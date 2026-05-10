import logging
import time
from typing import Dict, List

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from ..context import Context
from ..prompts import REWRITE_PROMPT
from ..state import AgentState

logger = logging.getLogger(__name__)


class QueryRewriteOutput(BaseModel):
    rewritten_query: str = Field(description="The improved query optimized for retrieval")
    reasoning: str = Field(description="Brief explanation of the improvement")


async def ainvoke_rewrite_query_step(
    state: AgentState,
    runtime: Runtime[Context],
) -> Dict[str, str | List]:
    """
    Rewrite query node: improve the query to get better retrieval results.

    WHY rewrite instead of just retrying?
    If the grader said the retrieved docs were irrelevant, the original query
    likely didn't match the index well. An LLM can reformulate the query with
    different vocabulary, add domain-specific terms, or broaden/narrow the
    scope — all of which improve recall from BM25 + vector search.

    Temperature=0.3 (slightly higher than 0.0) gives the model room to be
    creative with phrasing while still staying deterministic enough to be
    consistent across runs.

    Fallback: if the LLM call fails, append generic ML terms to the original
    query — crude but better than retrying with the exact same text.
    """
    logger.info("NODE: rewrite_query")
    start = time.time()
    original = state.get("original_query") or state["messages"][0].content

    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        span = runtime.context.langfuse_tracer.create_span(
            trace=runtime.context.trace,
            name="query_rewriting",
            input_data={"original_query": original, "attempt": state.get("retrieval_attempts", 0)},
        )

    rewritten = original
    reasoning = ""
    try:
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.3,
        )
        structured_llm = llm.with_structured_output(QueryRewriteOutput)
        result: QueryRewriteOutput = await structured_llm.ainvoke(
            REWRITE_PROMPT.format(question=original)
        )
        if result and result.rewritten_query.strip():
            rewritten = result.rewritten_query.strip()
            reasoning = result.reasoning
            logger.info(f"Query rewritten: '{original[:50]}' → '{rewritten[:50]}'")
        else:
            raise ValueError("Empty rewritten query")

    except Exception as e:
        logger.error(f"Query rewrite LLM failed, using keyword fallback: {e}")
        rewritten = f"{original} research paper arxiv machine learning"
        reasoning = f"Keyword fallback (LLM failed): {e}"

    if span:
        runtime.context.langfuse_tracer.end_span(
            span,
            output={"rewritten_query": rewritten, "reasoning": reasoning},
            metadata={"execution_time_ms": round((time.time() - start) * 1000, 2)},
        )

    return {
        "messages": [HumanMessage(content=rewritten)],
        "rewritten_query": rewritten,
    }
