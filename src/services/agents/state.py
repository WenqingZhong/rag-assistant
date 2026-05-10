from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .models import GradingResult, GuardrailScoring, RoutingDecision, SourceItem, ToolArtefact


class AgentState(dict):
    """
    Shared state threaded through every node in the LangGraph workflow.

    WHY TypedDict-style with add_messages?
    LangGraph passes a copy of this dict to each node and merges the node's
    return value back into it. The `add_messages` annotation on `messages`
    tells LangGraph to *append* new messages rather than *overwrite* the list —
    so when the retrieve node adds a tool-call message and tool_retrieve adds
    a tool-result message, both are preserved in order.

    All other fields use the default "last write wins" merge strategy.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: Optional[str]
    rewritten_query: Optional[str]
    retrieval_attempts: int
    guardrail_result: Optional[GuardrailScoring]
    routing_decision: Optional[RoutingDecision]
    sources: Optional[Dict[str, Any]]
    relevant_sources: List[SourceItem]
    relevant_tool_artefacts: Optional[List[ToolArtefact]]
    grading_results: List[GradingResult]
    metadata: Dict[str, Any]
