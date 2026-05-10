import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """
    Thin wrapper around the Langfuse SDK.

    WHY a wrapper instead of using the Langfuse SDK directly in route handlers?
    1. If Langfuse is disabled (no API keys) or its server is down, every SDK call
       would raise an exception and crash the request. The wrapper catches all errors
       and silently returns None — tracing failures never affect the user.
    2. All Langfuse SDK calls are in one place. Swapping the SDK version or
       switching to a different observability tool only requires changing this file.

    HOW Langfuse works:
    A "trace" is one top-level unit of work — e.g. one RAG request.
    A "span" is a named sub-step within that trace — e.g. "query_embedding",
    "search_retrieval", "llm_generation". Langfuse records the input, output,
    and duration of each span. You can then see the full pipeline breakdown
    in the Langfuse UI at http://localhost:3000.
    """

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=self.settings.public_key,
                    secret_key=self.settings.secret_key,
                    host=self.settings.host,
                    flush_at=self.settings.flush_at,
                    flush_interval=self.settings.flush_interval,
                    debug=self.settings.debug,
                )
                logger.info(f"Langfuse tracing enabled (host: {self.settings.host})")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.client = None
        else:
            logger.info("Langfuse tracing disabled — no credentials provided")

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager that opens a top-level trace for one RAG request.

        Usage:
            with langfuse_tracer.trace_rag_request(query="...") as trace:
                # all spans created inside here are children of this trace

        Yields the trace object (or None if Langfuse is disabled/errored).
        All spans must be created with trace.trace_id to be linked here.
        """
        if not self.client:
            yield None
            return

        try:
            trace = self.client.trace(
                name="rag_request",
                input={"query": query},
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
            )
            yield trace
        except Exception as e:
            logger.error(f"Error creating Langfuse trace: {e}")
            yield None

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a named span (sub-step) inside an existing trace.

        Returns None silently if Langfuse is disabled — callers never need
        to check: update_span() and span.end() are no-ops on None.
        """
        if not trace or not self.client:
            return None
        try:
            return self.client.span(
                trace_id=trace.trace_id,
                name=name,
                input=input_data,
            )
        except Exception as e:
            logger.error(f"Error creating span '{name}': {e}")
            return None

    def update_span(self, span, output: Any = None, metadata: Optional[Dict] = None):
        """Attach output data to a span before ending it."""
        if not span:
            return
        try:
            if output is not None:
                span.update(output=output)
            if metadata:
                span.update(metadata=metadata)
        except Exception as e:
            logger.error(f"Error updating span: {e}")

    def flush(self):
        """Push buffered events to Langfuse immediately (called after each request)."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Flush and close the connection — called on app shutdown."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")
