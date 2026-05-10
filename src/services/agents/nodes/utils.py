import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..models import SourceItem, ToolArtefact

logger = logging.getLogger(__name__)


def get_latest_query(messages: List) -> str:
    """Return the most recent HumanMessage content — the active query."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Return the most recent ToolMessage content — the retrieved documents."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content if hasattr(msg, "content") else ""
    return ""


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Build ToolArtefact records from all ToolMessages in history."""
    return [
        ToolArtefact(
            tool_name=getattr(msg, "name", "unknown"),
            tool_call_id=getattr(msg, "tool_call_id", ""),
            content=msg.content,
            metadata={},
        )
        for msg in messages
        if isinstance(msg, ToolMessage)
    ]
