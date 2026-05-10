from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GuardrailScoring(BaseModel):
    """
    LLM output for the guardrail node.

    The guardrail node asks the LLM: "Is this query about CS/AI/ML research?"
    and gets back a score (0-100) and a reason. If score < threshold (default 60),
    the request is routed to out_of_scope instead of retrieval.
    """
    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")


class GradeDocuments(BaseModel):
    """
    LLM output for the grade_documents node.

    After retrieval, the LLM grades whether the returned documents are actually
    relevant to the query. 'yes' routes to generate_answer; 'no' routes to
    rewrite_query for another retrieval attempt.
    """
    binary_score: Literal["yes", "no"] = Field(description="Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")


class SourceItem(BaseModel):
    """A single source paper returned in the agentic response."""
    arxiv_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    url: str
    relevance_score: float = Field(default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }


class ToolArtefact(BaseModel):
    """Metadata record of one tool call execution."""
    tool_name: str
    tool_call_id: str
    content: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Routing decision emitted by conditional edge functions."""
    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"]
    reason: str = Field(default="")


class GradingResult(BaseModel):
    """Structured record of one document grading outcome."""
    document_id: str
    is_relevant: bool
    score: float = Field(default=0.0)
    reasoning: str = Field(default="")
