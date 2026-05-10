import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.schemas.api.ask import AskRequest, AskResponse
from src.services.cache.client import CacheClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.langfuse.tracer import RAGTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ask"])


# ── Dependency providers ───────────────────────────────────────────────────────

def get_opensearch_client(request: Request) -> OpenSearchClient:
    return request.app.state.hybrid_opensearch_client


def get_jina_client(request: Request) -> JinaEmbeddingsClient:
    return request.app.state.jina_client


def get_ollama_client(request: Request) -> OllamaClient:
    return request.app.state.ollama_client


def get_langfuse_tracer(request: Request) -> LangfuseTracer:
    return request.app.state.langfuse_tracer


def get_cache_client(request: Request) -> Optional[CacheClient]:
    # Returns None if Redis was unavailable at startup — callers must handle None.
    return getattr(request.app.state, "cache_client", None)


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _retrieve_chunks(
    body: AskRequest,
    opensearch_client: OpenSearchClient,
    jina_client: JinaEmbeddingsClient,
    rag_tracer: RAGTracer,
    trace,
) -> tuple[list, str]:
    """
    Embed the query (if hybrid) and retrieve top_k chunks — with tracing.

    Each sub-step (embedding, search) is wrapped in its own Langfuse span so
    you can see in the dashboard exactly how long each step took.
    If Jina fails, falls back to BM25 and records the failure on the span.
    """
    query_embedding = None
    search_mode = "bm25"

    if body.use_hybrid:
        with rag_tracer.trace_embedding(trace, body.query) as embedding_span:
            try:
                query_embedding = await jina_client.embed_query(body.query)
                search_mode = "hybrid"
            except Exception as e:
                logger.warning(f"Jina embedding failed, falling back to BM25: {e}")
                rag_tracer.tracer.update_span(
                    embedding_span, output={"success": False, "error": str(e)}
                )

    with rag_tracer.trace_search(trace, body.query, body.top_k) as search_span:
        raw_results = opensearch_client.search_unified(
            query=body.query,
            query_embedding=query_embedding,
            size=body.top_k,
            categories=body.categories,
            use_hybrid=body.use_hybrid,
        )
        chunks = raw_results.get("hits", [])
        arxiv_ids = [c.get("arxiv_id", "") for c in chunks]
        rag_tracer.end_search(search_span, chunks, arxiv_ids, raw_results.get("total", 0))

    return chunks, search_mode


def _build_sources(chunks: list) -> list[str]:
    """Build deduplicated PDF URLs from chunk arxiv_ids."""
    seen: set = set()
    sources = []
    for chunk in chunks:
        arxiv_id = chunk.get("arxiv_id", "")
        if arxiv_id:
            clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            if url not in seen:
                sources.append(url)
                seen.add(url)
    return sources


# ── POST /api/v1/ask  (non-streaming) ─────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    opensearch_client: OpenSearchClient = Depends(get_opensearch_client),
    jina_client: JinaEmbeddingsClient = Depends(get_jina_client),
    ollama_client: OllamaClient = Depends(get_ollama_client),
    langfuse_tracer: LangfuseTracer = Depends(get_langfuse_tracer),
    cache_client: Optional[CacheClient] = Depends(get_cache_client),
) -> AskResponse:
    """
    Full RAG pipeline: retrieve chunks → generate answer → return structured response.

    New in week 6:
    - Cache check at the start: if the exact same request was made recently,
      return the stored answer immediately (skips Jina + OpenSearch + Ollama).
    - Langfuse tracing: each pipeline stage is recorded as a span so you can
      inspect timing, inputs, and outputs at http://localhost:3000.
    - Cache store at the end: save the answer for future identical requests.
    """
    if not opensearch_client.health_check():
        raise HTTPException(status_code=503, detail="Search service unavailable")

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", body.query) as trace:
        # ── Cache check ───────────────────────────────────────────────────────
        # O(1) Redis GET — if hit, skip the entire pipeline (~15-20s saved).
        if cache_client:
            cached = await cache_client.find_cached_response(body)
            if cached:
                logger.info(f"Cache HIT — returning stored answer for: '{body.query[:60]}'")
                return cached

        # ── Retrieve ──────────────────────────────────────────────────────────
        chunks, search_mode = await _retrieve_chunks(
            body, opensearch_client, jina_client, rag_tracer, trace
        )

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found for this query")

        sources = _build_sources(chunks)

        # ── Generate ──────────────────────────────────────────────────────────
        from src.services.ollama.prompts import RAGPromptBuilder
        prompt = RAGPromptBuilder().create_rag_prompt(body.query, chunks)

        with rag_tracer.trace_generation(trace, body.model, prompt) as gen_span:
            rag_result = await ollama_client.generate_rag_answer(
                query=body.query,
                chunks=chunks,
                model=body.model,
            )
            answer = rag_result.get("answer", "")
            rag_tracer.end_generation(gen_span, answer, body.model)

        rag_tracer.end_request(trace, answer, time.time() - start_time)

        response = AskResponse(
            query=body.query,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            search_mode=search_mode,
        )

        # ── Cache store ───────────────────────────────────────────────────────
        if cache_client:
            await cache_client.store_response(body, response)

        return response


# ── POST /api/v1/stream  (streaming SSE) ──────────────────────────────────────

@router.post("/stream")
async def stream(
    body: AskRequest,
    opensearch_client: OpenSearchClient = Depends(get_opensearch_client),
    jina_client: JinaEmbeddingsClient = Depends(get_jina_client),
    ollama_client: OllamaClient = Depends(get_ollama_client),
    langfuse_tracer: LangfuseTracer = Depends(get_langfuse_tracer),
    cache_client: Optional[CacheClient] = Depends(get_cache_client),
) -> StreamingResponse:
    """
    Streaming RAG with tracing and caching.

    Cache HIT path: stream the cached answer word-by-word (so the UI still
    sees the familiar token-by-token experience) then send the sources event.

    Cache MISS path: same as before — stream Ollama tokens live, then store
    the full assembled answer in Redis after generation completes.
    """
    if not opensearch_client.health_check():
        async def error_stream():
            yield f'data: {json.dumps({"error": "Search service unavailable", "done": True})}\n\n'
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def token_generator():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", body.query) as trace:
            # ── Cache check ───────────────────────────────────────────────────
            if cache_client:
                cached = await cache_client.find_cached_response(body)
                if cached:
                    logger.info(f"Cache HIT (stream) for: '{body.query[:60]}'")
                    # Re-stream cached answer word-by-word so the UI behaves identically
                    for word in cached.answer.split():
                        yield f'data: {json.dumps({"response": word + " ", "done": False})}\n\n'
                    yield f'data: {json.dumps({"response": "", "done": True})}\n\n'
                    yield f'data: {json.dumps({"type": "sources", "sources": cached.sources, "chunks_used": cached.chunks_used, "search_mode": cached.search_mode})}\n\n'
                    return

            # ── Retrieve ──────────────────────────────────────────────────────
            chunks, search_mode = await _retrieve_chunks(
                body, opensearch_client, jina_client, rag_tracer, trace
            )
            sources = _build_sources(chunks)

            if not chunks:
                yield f'data: {json.dumps({"error": "No relevant chunks found", "done": True})}\n\n'
                return

            # ── Stream generation ─────────────────────────────────────────────
            from src.services.ollama.prompts import RAGPromptBuilder
            prompt = RAGPromptBuilder().create_rag_prompt(body.query, chunks)

            full_answer = ""
            with rag_tracer.trace_generation(trace, body.model, prompt) as gen_span:
                async for token_dict in ollama_client.generate_rag_answer_stream(
                    query=body.query,
                    chunks=chunks,
                    model=body.model,
                ):
                    if token_dict.get("response"):
                        full_answer += token_dict["response"]
                    yield f"data: {json.dumps(token_dict)}\n\n"

                rag_tracer.end_generation(gen_span, full_answer, body.model)

            yield f'data: {json.dumps({"type": "sources", "sources": sources, "chunks_used": len(chunks), "search_mode": search_mode})}\n\n'

            rag_tracer.end_request(trace, full_answer, time.time() - start_time)

            # ── Cache store ───────────────────────────────────────────────────
            if cache_client and full_answer:
                await cache_client.store_response(
                    body,
                    AskResponse(
                        query=body.query,
                        answer=full_answer,
                        sources=sources,
                        chunks_used=len(chunks),
                        search_mode=search_mode,
                    ),
                )

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
