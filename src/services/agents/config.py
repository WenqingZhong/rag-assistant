from typing import Any, Dict

from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class GraphConfig(BaseModel):
    """
    Immutable configuration for one AgenticRAGService instance.

    Built once at startup and shared across all requests — nodes read values
    from this via the runtime Context rather than re-reading settings per call.

    Key thresholds:
    - guardrail_threshold=60: queries scoring below 60/100 are out-of-scope.
      Calibrated so "What is machine learning?" (borderline) still gets through
      while "What is a dog?" (clearly off-topic) is rejected.
    - max_retrieval_attempts=2: after two failed retrievals (docs graded as not
      relevant) the agent falls back to a graceful "couldn't find papers" message
      rather than looping indefinitely.
    """
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
    model: str = "llama3.2:1b"
    temperature: float = 0.0
    top_k: int = 3
    use_hybrid: bool = True
    enable_tracing: bool = True
    metadata: Dict[str, Any] = {}
    settings: Settings = Field(default_factory=get_settings)

    model_config = {"arbitrary_types_allowed": True}
