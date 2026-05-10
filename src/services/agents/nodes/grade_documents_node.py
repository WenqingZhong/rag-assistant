import logging
import time
from typing import Dict

from langgraph.runtime import Runtime

from ..context import Context
from ..models import GradeDocuments, GradingResult
from ..prompts import GRADE_DOCUMENTS_PROMPT
from ..state import AgentState
from .utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)


async def ainvoke_grade_documents_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict:
    """
    Grade documents node: decide if retrieved docs are relevant.

    WHY grade at all?
    A keyword/vector search can return documents that score well against the
    query embedding but don't actually answer the question. Grading catches
    these misses before they propagate to generation, where a hallucinated
    answer would be worse than saying "I couldn't find relevant papers."

    Routing:
    - 'yes' (relevant)  → generate_answer
    - 'no'  (not relevant) → rewrite_query (up to max_retrieval_attempts)

    Fallback: if the LLM grading call fails, fall back to a heuristic —
    "is there enough text?" — rather than blocking the request.
    """
    logger.info("NODE: grade_documents")
    start = time.time()
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        span = runtime.context.langfuse_tracer.create_span(
            trace=runtime.context.trace,
            name="document_grading",
            input_data={"query": question, "context_length": len(context)},
        )

    if not context:
        logger.warning("No retrieved context — routing to rewrite_query")
        if span:
            runtime.context.langfuse_tracer.end_span(
                span,
                output={"routing_decision": "rewrite_query", "reason": "no_context"},
                metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
            )
        return {"routing_decision": "rewrite_query", "grading_results": []}

    try:
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0,
        )
        structured_llm = llm.with_structured_output(GradeDocuments)
        grading: GradeDocuments = await structured_llm.ainvoke(
            GRADE_DOCUMENTS_PROMPT.format(context=context, question=question)
        )
        is_relevant = grading.binary_score == "yes"
        logger.info(f"Grading — relevant={is_relevant}, reasoning={grading.reasoning}")

    except Exception as e:
        logger.error(f"Grading LLM failed, using heuristic: {e}")
        is_relevant = len(context.strip()) > 50
        grading = GradeDocuments(
            binary_score="yes" if is_relevant else "no",
            reasoning=f"Heuristic fallback (LLM failed): {e}",
        )

    route = "generate_answer" if is_relevant else "rewrite_query"
    result = GradingResult(
        document_id="retrieved_docs",
        is_relevant=is_relevant,
        score=1.0 if is_relevant else 0.0,
        reasoning=grading.reasoning,
    )

    if span:
        runtime.context.langfuse_tracer.end_span(
            span,
            output={"routing_decision": route, "is_relevant": is_relevant},
            metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
        )

    return {"routing_decision": route, "grading_results": [result]}
