import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """
    Thin wrapper around the Langfuse v3 SDK.

    WHY v3?
    Week 7 introduces agentic RAG via LangGraph. Langfuse v3 ships a
    CallbackHandler that auto-traces every LLM call inside a LangGraph graph
    without any manual span management — you just pass the handler as a
    LangGraph callback and every node's LLM call is recorded automatically.
    v2 had no LangGraph integration.

    HOW v3 context propagation works:
    In v3, child spans don't need an explicit trace_id. When you open a
    start_as_current_span() context, all spans created inside that context
    are automatically linked as children via thread-local state. This means
    create_span() no longer needs to extract trace.trace_id — it just calls
    self.client.span() and the SDK wires the parent-child link itself.

    All methods silently no-op when self.client is None (Langfuse disabled
    or credentials missing), so callers never need to guard against None.
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
                logger.info(f"Langfuse v3 tracing enabled (host: {self.settings.host})")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
                self.client = None
        else:
            logger.info("Langfuse tracing disabled — no credentials provided")

    # ── Non-agentic tracing (used by /ask and /stream via RAGTracer) ──────────

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

        In Langfuse v2, client.trace() returns a StatefulTraceClient.
        Child spans are created by calling trace.span() on that object.
        """
        if not self.client:
            yield None
            return

        try:
            trace = self.client.trace(
                name="rag_request",
                input={"query": query},
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            yield trace
        except Exception as e:
            logger.error(f"Error creating Langfuse trace: {e}")
            yield None

    # ── Span helpers (used by both RAGTracer and agent nodes) ─────────────────

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a named child span on an existing trace (Langfuse v2 API).

        In v2, spans must be created on the trace object — the root Langfuse
        client has no .span() method. The trace arg is required (not optional).
        """
        if not self.client or trace is None:
            return None
        try:
            return trace.span(
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Error creating span '{name}': {e}")
            return None

    def end_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Attach output/metadata to a span and close it.

        Agent nodes call end_span() when a pipeline step finishes so the
        Langfuse dashboard shows the exact wall-clock duration of each step.
        """
        if not span:
            return
        try:
            update_kwargs: Dict[str, Any] = {}
            if output is not None:
                update_kwargs["output"] = output
            if metadata is not None:
                update_kwargs["metadata"] = metadata
            if update_kwargs:
                span.update(**update_kwargs)
            span.end()
        except Exception as e:
            logger.error(f"Error ending span: {e}")

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """Update a span without closing it (used mid-step to attach partial data)."""
        if not span:
            return
        try:
            update_data: Dict[str, Any] = {}
            if output is not None:
                update_data["output"] = output
            if metadata is not None:
                update_data["metadata"] = metadata
            if level is not None:
                update_data["level"] = level
            if status_message is not None:
                update_data["status_message"] = status_message
            if update_data:
                span.update(**update_data)
        except Exception as e:
            logger.error(f"Error updating span: {e}")

    # ── Agentic / LangChain integration ───────────────────────────────────────

    def start_trace(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a top-level Langfuse trace via the v2 REST API.

        Returns the trace object so callers can attach output and spans.
        Returns None when Langfuse is disabled.
        """
        if not self.client:
            return None
        try:
            return self.client.trace(
                name=name,
                input=input_data or {},
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning(f"Failed to create Langfuse trace: {e}")
            return None

    def submit_feedback(
        self,
        trace_id: str,
        score: float,
        name: str = "user-feedback",
        comment: Optional[str] = None,
    ) -> bool:
        """
        Attach a user feedback score to a completed trace.

        Called by POST /feedback so you can correlate answer quality with
        the retrieval and generation steps visible in the Langfuse dashboard.
        """
        if not self.client:
            logger.warning("Cannot submit feedback: Langfuse is disabled")
            return False
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment,
            )
            logger.info(f"Submitted feedback for trace {trace_id}: score={score}")
            return True
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def flush(self):
        """Push buffered events to Langfuse immediately."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse: {e}")

    def shutdown(self):
        """Flush and close — called on app shutdown."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")
