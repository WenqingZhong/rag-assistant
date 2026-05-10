from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request body for the RAG question-answering endpoints
    (POST /api/v1/ask and POST /api/v1/stream).

    WHY top_k instead of size?
    In the hybrid search endpoint we return many hits for the user to browse.
    In the RAG endpoint the hits are never shown directly — they're fed to the
    LLM as context. More chunks = longer prompt = slower generation + higher
    risk of the LLM losing focus. 3 chunks is enough context for a focused
    answer; 10 is the practical ceiling for a 1B-parameter model like llama3.2:1b.

    WHY expose `model` in the request?
    Different questions benefit from different models. A simple factual question
    works fine with llama3.2:1b (fast, low memory). A nuanced comparison across
    papers benefits from a larger model if available. Exposing the field lets
    the UI or caller choose per-request without restarting the API.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to answer using retrieved paper chunks",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of chunks to retrieve and pass to the LLM as context",
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid search (BM25 + vector). False = BM25 only.",
    )
    model: str = Field(
        default="llama3.2:1b",
        description="Ollama model to use for answer generation",
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter retrieved chunks to these arXiv categories (e.g. ['cs.AI'])",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are transformers in machine learning?",
                "top_k": 3,
                "use_hybrid": True,
                "model": "llama3.2:1b",
                "categories": ["cs.AI", "cs.LG"],
            }
        }
    }


class AskResponse(BaseModel):
    """
    Response from the non-streaming RAG endpoint (POST /api/v1/ask).

    AskResponse
        query       ← the original question (echoed back)
        answer      ← taken from RAGResponse.answer
        sources     ← taken from RAGResponse.sources (PDF URLs generated from the arxiv_ids of retrieved chunks.)
        chunks_used ← how many chunks were retrieved (not in RAGResponse)
        search_mode ← "hybrid" or "bm25" (not in RAGResponse) — tells the caller whether semantic search
                 was used. Useful for debugging when answers seem off-topic. 
    """

    query: str = Field(..., description="The original question")
    answer: str = Field(..., description="LLM-generated answer based on retrieved chunks")
    sources: List[str] = Field(..., description="PDF URLs of papers used as context")
    chunks_used: int = Field(..., description="Number of chunks passed to the LLM")
    search_mode: str = Field(..., description="Search mode used: 'hybrid' or 'bm25'")

    #json_schema_extra with an "example" key tells FastAPI what 
    # to pre-fill in the interactive docs at http://localhost:8000/docs
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture based on self-attention...",
                "sources": [
                    "https://arxiv.org/pdf/1706.03762.pdf",
                    "https://arxiv.org/pdf/1810.04805.pdf",
                ],
                "chunks_used": 3,
                "search_mode": "hybrid",
            }
        }
    }


class AgenticAskResponse(BaseModel):
    """
    Response from the agentic RAG endpoint (POST /api/v1/ask-agentic).

    Extends AskResponse with three agentic-specific fields:

    reasoning_steps — human-readable log of what the agent decided at each
        stage (guardrail score, retrieval attempts, grading outcome). Lets
        the caller understand WHY the agent produced this answer.

    retrieval_attempts — how many retrieve → grade cycles ran before the
        agent either found relevant docs or gave up. 1 = first try worked;
        2 = query was rewritten once.

    trace_id — Langfuse trace ID for this request. The caller can POST this
        to /feedback with a score so engineers can correlate answer quality
        with the pipeline steps visible in the Langfuse dashboard.
    """
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    chunks_used: int
    search_mode: str
    reasoning_steps: List[str] = Field(default_factory=list)
    retrieval_attempts: int = Field(default=0)
    trace_id: Optional[str] = Field(default=None)


class FeedbackRequest(BaseModel):
    """Request body for POST /api/v1/feedback."""
    trace_id: str = Field(..., description="Langfuse trace ID from the AgenticAskResponse")
    score: float = Field(..., ge=-1, le=1, description="Feedback score: 1=good, -1=bad, 0=neutral")
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    """Response from POST /api/v1/feedback."""
    success: bool
    message: str
