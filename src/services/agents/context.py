from dataclasses import dataclass
from typing import Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient


@dataclass
class Context:
    """
    Immutable runtime dependencies injected into every agent node.

    WHY a separate Context instead of closures?
    LangGraph nodes are plain functions (or coroutines). Passing dependencies
    via closures would mean re-creating the graph on every request. Instead,
    we build the graph once and inject a fresh Context at invocation time via
    LangGraph's `context=` parameter. Nodes access it through Runtime[Context].

    This also makes nodes trivially testable — just construct a Context with
    mocked clients and invoke the node function directly.
    """
    ollama_client: OllamaClient
    opensearch_client: OpenSearchClient
    embeddings_client: JinaEmbeddingsClient
    langfuse_tracer: Optional[LangfuseTracer]
    trace: Optional[object] = None          # active Langfuse span (if tracing enabled)
    langfuse_enabled: bool = False
    model_name: str = "llama3.2:1b"
    temperature: float = 0.0
    top_k: int = 3
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
