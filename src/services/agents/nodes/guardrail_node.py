import logging
import time
from typing import Dict, Literal

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GuardrailScoring
from ..prompts import GUARDRAIL_PROMPT
from ..state import AgentState
from .utils import get_latest_query

logger = logging.getLogger(__name__)


def continue_after_guardrail(
    state: AgentState,
    runtime: Runtime[Context],
) -> Literal["continue", "out_of_scope"]:
    """
    Conditional edge: route based on guardrail score.

    Called by LangGraph after the guardrail node completes. Returns "continue"
    (→ retrieve) if the score meets the threshold, "out_of_scope" otherwise.
    The threshold is read from runtime.context so it can be tuned per-request
    without rebuilding the graph.
    """
    guardrail_result = state.get("guardrail_result")
    if not guardrail_result:
        logger.warning("No guardrail result — defaulting to continue")
        return "continue"

    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold
    decision = "continue" if score >= threshold else "out_of_scope"
    logger.info(f"Guardrail score={score}, threshold={threshold} → {decision}")
    return decision


async def ainvoke_guardrail_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict:
    """
    Guardrail node: score the query's relevance to CS/AI/ML research.

    WHY a guardrail?
    Without it, the agent would always try to retrieve papers — even for
    "What is 2+2?" or "Hello". Sending out-of-scope queries to OpenSearch
    wastes latency and produces nonsensical answers. The LLM scores on a
    0-100 scale with a structured output schema so the score is always a
    valid integer, never free-text.

    Fallback: if the LLM call fails, score defaults to 50 (borderline)
    so the agent continues rather than silently blocking all requests.
    """
    logger.info("NODE: guardrail")
    start = time.time()
    query = get_latest_query(state["messages"])

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        span = runtime.context.langfuse_tracer.create_span(
            trace=runtime.context.trace,
            name="guardrail_validation",
            input_data={"query": query, "threshold": runtime.context.guardrail_threshold},
        )

    try:
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(GuardrailScoring)
        response: GuardrailScoring = await structured_llm.ainvoke(
            GUARDRAIL_PROMPT.format(question=query)
        )
        logger.info(f"Guardrail — score={response.score}, reason={response.reason}")

        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"score": response.score, "reason": response.reason},
                metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
            )

    except Exception as e:
        logger.error(f"Guardrail LLM failed, using fallback score=50: {e}")
        response = GuardrailScoring(score=50, reason=f"LLM failed: {e}")
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"score": 50, "error": str(e)},
                metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
            )

    return {"guardrail_result": response}
