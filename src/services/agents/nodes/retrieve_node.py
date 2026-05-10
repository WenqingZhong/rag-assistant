import logging
import time
from typing import Dict, Union

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """
    Retrieve node: create a tool call or emit a fallback after max attempts.

    WHY does this node not do the actual retrieval?
    LangGraph separates *requesting* a tool call from *executing* it.
    This node emits an AIMessage with a `tool_calls` field pointing at
    `retrieve_papers`. LangGraph then routes to `tool_retrieve` (a built-in
    ToolNode) which executes the tool and appends the ToolMessage result.
    This separation lets LangGraph handle concurrency, error recovery, and
    tracing for tool calls automatically.

    Max-attempts guard:
    If retrieval_attempts >= max_retrieval_attempts, we've already tried
    retrieval and query-rewriting once. A third attempt would likely produce
    the same poor result, so we return a graceful fallback message instead.
    """
    logger.info("NODE: retrieve")
    start = time.time()
    messages = state["messages"]
    question = get_latest_query(messages)
    current_attempts = state.get("retrieval_attempts", 0)
    max_attempts = runtime.context.max_retrieval_attempts

    updates: Dict = {}

    if state.get("original_query") is None:
        updates["original_query"] = question

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        span = runtime.context.langfuse_tracer.create_span(
            trace=runtime.context.trace,
            name="document_retrieval_initiation",
            input_data={"query": question, "attempt": current_attempts + 1, "max_attempts": max_attempts},
        )

    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached")
        fallback = (
            f"I couldn't find relevant research papers after {max_attempts} attempt(s).\n"
            "This may mean no matching papers are indexed or the query terms don't match.\n"
            "Try rephrasing with more specific technical terms."
        )
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"status": "max_attempts_reached"},
                metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
            )
        return {**updates, "messages": [AIMessage(content=fallback)]}

    new_count = current_attempts + 1
    updates["retrieval_attempts"] = new_count
    logger.info(f"Retrieval attempt {new_count}/{max_attempts} for query: {question[:80]}")

    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[{
                "id": f"retrieve_{new_count}",
                "name": "retrieve_papers",
                "args": {"query": question},
            }],
        )
    ]

    if span:
        runtime.context.langfuse_tracer.end_span(
            span,
            output={"status": "tool_call_created", "attempt": new_count},
            metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
        )

    return updates
