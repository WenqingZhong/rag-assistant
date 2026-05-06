"""Pydantic models for Ollama structured outputs."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """
    Structured output that Ollama is asked to produce for every RAG query.

    WHY ask Ollama for structured JSON instead of free-form text?
    Free-form text is hard to parse reliably — the LLM might put sources
    inline, omit them, or change format across runs. Asking for JSON with
    a fixed schema lets us validate the response with Pydantic and guarantee
    the API always returns the same fields, even if the LLM output is imperfect.

    The flow:
        1. We send Ollama the JSON schema of this model as part of the prompt.
        2. Ollama generates a JSON string that matches the schema.
        3. ResponseParser.parse_structured_response() validates it here.
        4. If validation fails, the parser falls back to treating the whole
           output as plain text and populating `answer` only.

    Fields:
        answer:     The actual answer text — what the user reads.
        sources:    PDF URLs extracted from the retrieved chunks. Populated
                    by OllamaClient after parsing, not by the LLM itself
                    (LLMs hallucinate URLs; we generate them from arxiv_ids).
        confidence: Self-reported confidence level from the LLM.
                    "high"   = multiple chunks clearly support the answer.
                    "medium" = some relevant chunks found, partial support.
                    "low"    = chunks found but weakly relevant.
                    Optional — not all models reliably produce this.
        citations:  arXiv IDs the LLM explicitly referenced in its answer.
                    e.g. ["2301.07041", "1706.03762"].
                    Optional — useful for showing inline citations in the UI.
    """

    answer: str = Field(
        description="Comprehensive answer based on the provided paper excerpts"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="PDF URLs of source papers used in the answer",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: high, medium, or low based on excerpt relevance",
    )
    citations: Optional[List[str]] = Field(
        default=None,
        description="arXiv IDs referenced in the answer (e.g. ['2301.07041'])",
    )
