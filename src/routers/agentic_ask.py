import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.schemas.api.ask import (
    AgenticAskResponse,
    AskRequest,
    FeedbackRequest,
    FeedbackResponse,
)
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.langfuse.client import LangfuseTracer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["agentic-rag"])


# ── Dependency providers ───────────────────────────────────────────────────────

def get_agentic_rag_service(request: Request) -> AgenticRAGService:
    return request.app.state.agentic_rag_service


def get_langfuse_tracer(request: Request) -> Optional[LangfuseTracer]:
    return getattr(request.app.state, "langfuse_tracer", None)


# ── POST /api/v1/ask-agentic ──────────────────────────────────────────────────

@router.post("/ask-agentic", response_model=AgenticAskResponse)
async def ask_agentic(
    body: AskRequest,
    agentic_rag: AgenticRAGService = Depends(get_agentic_rag_service),
) -> AgenticAskResponse:
    """
    Agentic RAG endpoint with intelligent retrieval and query refinement.

    Unlike POST /ask (which always retrieves → generates), this endpoint
    runs a LangGraph workflow that:

    1. Guardrail  — scores query relevance (0-100). Rejects off-topic queries.
    2. Retrieve   — searches OpenSearch via the retrieve_papers tool.
    3. Grade      — LLM judges whether retrieved docs actually answer the question.
    4. Rewrite    — if grading fails, rewrites the query and retries (max 2x).
    5. Generate   — produces the final answer from relevant context.

    The response includes reasoning_steps so callers can see what the agent
    decided at each stage, and a trace_id for submitting feedback via /feedback.
    """
    try:
        result = await agentic_rag.ask(query=body.query)

        return AgenticAskResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result.get("sources", []),
            chunks_used=body.top_k,
            search_mode="hybrid" if body.use_hybrid else "bm25",
            reasoning_steps=result.get("reasoning_steps", []),
            retrieval_attempts=result.get("retrieval_attempts", 0),
            trace_id=result.get("trace_id"),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Agentic RAG error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")


# ── POST /api/v1/feedback ─────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    body: FeedbackRequest,
    langfuse_tracer: Optional[LangfuseTracer] = Depends(get_langfuse_tracer),
) -> FeedbackResponse:
    """
    Attach user feedback to a completed agentic RAG trace.

    WHY: The trace_id in AgenticAskResponse links back to the full pipeline
    trace in Langfuse (guardrail score, retrieval attempts, grading result,
    generated answer). Attaching a user score to that trace lets you correlate
    pipeline behaviour with perceived answer quality in the Langfuse dashboard.

    Score convention: 1.0 = good answer, 0.0 = neutral, -1.0 = bad answer.
    """
    if not langfuse_tracer:
        raise HTTPException(
            status_code=503,
            detail="Langfuse tracing is disabled — cannot submit feedback.",
        )

    success = langfuse_tracer.submit_feedback(
        trace_id=body.trace_id,
        score=body.score,
        comment=body.comment,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to submit feedback to Langfuse")

    langfuse_tracer.flush()
    return FeedbackResponse(success=True, message="Feedback recorded successfully")
