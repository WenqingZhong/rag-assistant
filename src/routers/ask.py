import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.schemas.api.ask import AskRequest, AskResponse
from src.services.embeddings.jina_client import JinaEmbeddingsClient
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


# ── Shared retrieval helper ────────────────────────────────────────────────────

async def _retrieve_chunks(
    body: AskRequest,
    opensearch_client: OpenSearchClient,
    jina_client: JinaEmbeddingsClient,
) -> tuple[list, str]:
    """
    Embed the query (if hybrid) and retrieve top_k chunks from OpenSearch.

    Returns (chunks, search_mode).

    WHY embed here instead of inside each route?
    Both /ask and /stream do the same retrieval step. Putting it in one place
    means a change to retrieval logic only happens once.

    WHY graceful fallback instead of propagating the Jina error?
    If Jina is down, BM25 still returns useful results. A 503 error for
    the whole request is worse than a keyword-only answer.
    """
    query_embedding = None
    search_mode = "bm25"

    if body.use_hybrid:
        try:
            query_embedding = await jina_client.embed_query(body.query)
            search_mode = "hybrid"
        except Exception as e:
            logger.warning(f"Jina embedding failed, falling back to BM25: {e}")

    raw_results = opensearch_client.search_unified(
        query=body.query,
        query_embedding=query_embedding,
        size=body.top_k,
        categories=body.categories,
        use_hybrid=body.use_hybrid,
    )
    return raw_results.get("hits", []), search_mode


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
) -> AskResponse:
    """
    Full RAG pipeline: retrieve chunks → generate answer → return structured response.

    Uses Ollama's structured output (format=schema) so the response is always
    valid JSON. Waits for the complete answer before returning — use /stream
    if you want tokens as they arrive.

    Request body:
        {
            "query": "What is self-attention?",
            "top_k": 3,
            "use_hybrid": true,
            "model": "llama3.2:1b",
            "categories": ["cs.AI"]
        }

    Response:
        {
            "query": "What is self-attention?",
            "answer": "Self-attention is...",
            "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],
            "chunks_used": 3,
            "search_mode": "hybrid"
        }
    """
    if not opensearch_client.health_check():
        raise HTTPException(status_code=503, detail="Search service unavailable")

    chunks, search_mode = await _retrieve_chunks(body, opensearch_client, jina_client)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found for this query")

    logger.info(f"Generating RAG answer for: '{body.query[:60]}' ({len(chunks)} chunks, mode={search_mode})")

    rag_result = await ollama_client.generate_rag_answer(
        query=body.query,
        chunks=chunks,
        model=body.model,
    )

    return AskResponse(
        query=body.query,
        answer=rag_result["answer"],
        sources=rag_result.get("sources", []),
        chunks_used=len(chunks),
        search_mode=search_mode,
    )


# ── POST /api/v1/stream  (streaming SSE) ──────────────────────────────────────

@router.post("/stream")
async def stream(
    body: AskRequest,
    opensearch_client: OpenSearchClient = Depends(get_opensearch_client),
    jina_client: JinaEmbeddingsClient = Depends(get_jina_client),
    ollama_client: OllamaClient = Depends(get_ollama_client),
) -> StreamingResponse:
    """
    Streaming RAG: retrieve chunks → stream tokens from Ollama as Server-Sent Events.

    WHY SSE instead of WebSocket?
    SSE is one-directional (server → client) and works over plain HTTP/1.1.
    The browser sends one request and receives a stream of events — no need for
    the bidirectional channel that WebSocket adds. For token streaming, SSE is
    simpler and sufficient.

    WHAT the browser receives — two types of SSE events:

        1. Token events (one per Ollama token):
               data: {"response": "Trans", "done": false}
               data: {"response": "form", "done": false}
               data: {"response": "ers", "done": false}
               ...
               data: {"response": "", "done": true, "total_duration": 14823000000}

           The browser appends each "response" fragment to the displayed text.
           When "done" is true, token generation is complete.

        2. Sources event (one, after all tokens):
               data: {"type": "sources", "sources": [...], "chunks_used": 3, "search_mode": "hybrid"}

           The browser uses this to render the source links below the answer.

    WHY a separate sources event instead of embedding sources in the done token?
    Ollama's final token already has "done": true. Mutating it to add sources
    would tie our schema to Ollama's internal format. A separate named event
    keeps the two concerns independent and makes the browser logic simpler.

    SSE format: each event is "data: {json}\\n\\n" (two newlines end the event).
    The browser's EventSource API parses this automatically.

    Headers:
        Cache-Control: no-cache        — don't buffer the stream
        X-Accel-Buffering: no          — disable nginx/proxy buffering
        Connection: keep-alive         — keep the HTTP connection open
    """
    if not opensearch_client.health_check():
        # Can't raise HTTPException inside a StreamingResponse generator.
        # Return a single SSE error event with HTTP 200 so the browser receives it.
        async def error_stream():
            yield f'data: {json.dumps({"error": "Search service unavailable", "done": True})}\n\n'

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    chunks, search_mode = await _retrieve_chunks(body, opensearch_client, jina_client)
    sources = _build_sources(chunks)

    async def token_generator():
        if not chunks:
            yield f'data: {json.dumps({"error": "No relevant chunks found", "done": True})}\n\n'
            return

        logger.info(f"Streaming RAG for: '{body.query[:60]}' ({len(chunks)} chunks, mode={search_mode})")

        async for token_dict in ollama_client.generate_rag_answer_stream(
            query=body.query,
            chunks=chunks,
            model=body.model,
        ):
            # Each token_dict looks like: {"response": "Trans", "done": False}
            # or the final:              {"response": "", "done": True, "total_duration": ...}
            yield f"data: {json.dumps(token_dict)}\n\n"

        # After all tokens, send one metadata event with sources and search info.
        yield f'data: {json.dumps({"type": "sources", "sources": sources, "chunks_used": len(chunks), "search_mode": search_mode})}\n\n'

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
