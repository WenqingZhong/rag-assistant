import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .client import LangfuseTracer


class RAGTracer:
    """
    High-level tracing API for the non-agentic RAG pipeline (/ask, /stream).

    Wraps LangfuseTracer with pipeline-specific context managers so route
    handlers don't need to know about span names or input schemas.

    Each context manager opens a span on entry and closes it on exit via
    end_span(), capturing wall-clock duration automatically.

    USAGE IN A ROUTE:
        rag_tracer = RAGTracer(langfuse_tracer)

        with rag_tracer.trace_request("user_123", query) as trace:
            with rag_tracer.trace_embedding(trace, query) as span:
                embedding = await jina.embed_query(query)

            with rag_tracer.trace_search(trace, query, top_k) as span:
                chunks = opensearch.search_unified(...)
                rag_tracer.end_search(span, chunks, arxiv_ids, total)

            with rag_tracer.trace_generation(trace, model, prompt) as span:
                answer = await ollama.generate_rag_answer(...)
                rag_tracer.end_generation(span, answer, model)

            rag_tracer.end_request(trace, answer, elapsed)
    """

    def __init__(self, tracer: LangfuseTracer):
        self.tracer = tracer

    @contextmanager
    def trace_request(self, user_id: str, query: str):
        """Top-level trace for one complete RAG request."""
        with self.tracer.trace_rag_request(
            query=query,
            user_id=user_id,
            session_id=f"session_{user_id}",
        ) as trace:
            try:
                yield trace
            finally:
                if trace:
                    self.tracer.flush()

    @contextmanager
    def trace_embedding(self, trace, query: str):
        """Span for the Jina embed_query() call."""
        start = time.time()
        span = self.tracer.create_span(
            trace=trace,
            name="query_embedding",
            input_data={"query": query, "query_length": len(query)},
        )
        try:
            yield span
        finally:
            duration_ms = round((time.time() - start) * 1000, 2)
            self.tracer.end_span(
                span,
                output={"duration_ms": duration_ms, "success": True},
            )

    @contextmanager
    def trace_search(self, trace, query: str, top_k: int):
        """Span for the OpenSearch search_unified() call."""
        span = self.tracer.create_span(
            trace=trace,
            name="search_retrieval",
            input_data={"query": query, "top_k": top_k},
        )
        try:
            yield span
        finally:
            self.tracer.end_span(span)

    def end_search(self, span, chunks: List[Dict], arxiv_ids: List[str], total_hits: int):
        """Attach search result metadata to the search span before it closes."""
        self.tracer.update_span(
            span,
            output={
                "chunks_returned": len(chunks),
                "unique_papers": len(set(arxiv_ids)),
                "total_hits": total_hits,
                "arxiv_ids": list(set(arxiv_ids)),
            },
        )

    @contextmanager
    def trace_generation(self, trace, model: str, prompt: str):
        """Span for the Ollama LLM generation call."""
        span = self.tracer.create_span(
            trace=trace,
            name="llm_generation",
            input_data={
                "model": model,
                "prompt_length": len(prompt),
                "prompt": prompt,
            },
        )
        try:
            yield span
        finally:
            self.tracer.end_span(span)

    def end_generation(self, span, response: str, model: str):
        """Attach the LLM response to the generation span before it closes."""
        self.tracer.update_span(
            span,
            output={
                "response": response,
                "response_length": len(response),
                "model_used": model,
            },
        )

    def end_request(self, trace, answer: str, total_duration: float):
        """Attach the final answer to the top-level trace."""
        if not trace:
            return
        try:
            trace.update(
                output={
                    "answer": answer,
                    "total_duration_seconds": round(total_duration, 3),
                    "response_length": len(answer),
                }
            )
        except Exception:
            pass
