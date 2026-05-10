import logging
import time
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..context import Context
from ..prompts import GENERATE_ANSWER_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """
    Generate answer node: produce the final answer from retrieved context.

    This is the only node that generates a free-form answer (not structured
    output). Temperature=0.0 (deterministic) keeps answers reproducible for
    the same query+context pair.

    The prompt explicitly instructs the model to:
    - Use ONLY the retrieved papers (no hallucination)
    - Cite papers by title or arxiv ID
    - Acknowledge if the papers don't fully answer the question

    Fallback: if the LLM call fails, return an error message rather than
    crashing the request — the user still gets a response, just not a good one.
    """
    logger.info("NODE: generate_answer")
    start = time.time()
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"]) or "No relevant documents found."

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        span = runtime.context.langfuse_tracer.create_span(
            trace=runtime.context.trace,
            name="answer_generation",
            input_data={
                "query": question,
                "context_length": len(context),
                "sources_count": len(state.get("relevant_sources", [])),
            },
        )

    try:
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=runtime.context.temperature,
        )
        response = await llm.ainvoke(
            GENERATE_ANSWER_PROMPT.format(context=context, question=question)
        )
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Answer generated — {len(answer)} chars")

        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"answer_length": len(answer)},
                metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
            )

    except Exception as e:
        logger.error(f"Answer generation LLM failed: {e}")
        answer = f"I encountered an error while generating the answer: {e}\nPlease try again."
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"error": str(e)},
                metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
            )

    return {"messages": [AIMessage(content=answer)]}
