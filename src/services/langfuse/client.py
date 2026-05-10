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

        In v3, start_as_current_span() replaces the old client.trace() call.
        Any spans created inside this context are automatically linked as
        children — no explicit trace_id passing needed.
        """
        if not self.client:
            yield None
            return

        try:
            with self.client.start_as_current_span(name="rag_request") as span:
                span.update(
                    input={"query": query},
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata or {},
                )
                yield span
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
        Create a named child span inside the current trace context.

        The `trace` argument is kept for API compatibility with RAGTracer and
        the agent nodes, but in v3 it's only used as an "is tracing active?"
        check — the SDK links the span to the current parent automatically.
        """
        if not self.client or trace is None:
            return None
        try:
            return self.client.span(
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

    def get_callback_handler(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
    ):
        """
        Return a LangChain CallbackHandler for automatic LLM call tracing.

        When passed as a LangGraph callback, every LLM invocation inside the
        graph is recorded as a Langfuse generation span — model name, prompt,
        output, and token counts — with zero manual instrumentation.
        """
        if not self.client:
            return None
        try:
            from langfuse.langchain import CallbackHandler
            return CallbackHandler(
                trace_name=trace_name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                tags=tags,
            )
        except Exception as e:
            logger.error(f"Error creating CallbackHandler: {e}")
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
