import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from src.config import get_settings
import redis

from src.routers.agentic_ask import router as agentic_ask_router
from src.routers.ask import router as ask_router
from src.routers.documents import router as documents_router
from src.routers.hybrid_search import router as hybrid_search_router
from src.routers.search import router as search_router
from src.services.agents.agentic_rag import AgenticRAGService
from src.services.agents.config import GraphConfig
from src.schemas.api.health import HealthResponse, ServiceStatus
from src.services.cache.client import CacheClient
from src.services.database import create_tables, get_session
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama import OllamaClient
from src.services.openai.client import OpenAIClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG Assistant API...")
    settings = get_settings()

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    create_tables()
    logger.info("PostgreSQL tables ready")

    # ── OpenSearch (hybrid chunks index) ─────────────────────────────────────
    # A single OpenSearchClient instance handles both the existing arxiv-papers
    # index (via search_papers) and the new arxiv-papers-chunks index (via
    # bulk_index_chunks / search_unified). Stored under hybrid_opensearch_client
    # so hybrid_search.py's dependency provider can find it without ambiguity.
    hybrid_os_client = OpenSearchClient(host=settings.opensearch.host)
    app.state.hybrid_opensearch_client = hybrid_os_client

    if hybrid_os_client.health_check():
        # Create arxiv-papers-chunks index and register the RRF pipeline if
        # they don't exist yet. Idempotent — safe to call on every startup.
        setup = hybrid_os_client.setup_hybrid_indices(force=False)
        logger.info(
            f"Hybrid index {'created' if setup['hybrid_index'] else 'already exists'}, "
            f"RRF pipeline {'created' if setup['rrf_pipeline'] else 'already exists'}"
        )
    else:
        logger.warning("OpenSearch unavailable at startup — hybrid search will return 503")

    # ── Jina embeddings client ────────────────────────────────────────────────
    # Stored in app.state so the persistent httpx connection pool is reused
    # across all requests (one TLS handshake at startup, not per request).
    # Closed gracefully on shutdown to drain in-flight connections.
    jina_client = JinaEmbeddingsClient(api_key=settings.jina_api_key)
    app.state.jina_client = jina_client
    logger.info("Jina embeddings client ready")

    # ── LLM client (OpenAI or Ollama) ─────────────────────────────────────────
    # Both clients expose the same interface (generate_rag_answer,
    # generate_rag_answer_stream, get_langchain_model) so all routers and the
    # agentic service work unchanged regardless of which one is active.
    if settings.openai.enabled and settings.openai.api_key:
        llm_client = OpenAIClient(
            api_key=settings.openai.api_key,
            model=settings.openai.model,
            temperature=settings.openai.temperature,
        )
        app.state.ollama_client = llm_client
        logger.info(f"OpenAI client ready (model: {settings.openai.model})")
    else:
        llm_client = OllamaClient(
            base_url=settings.ollama_host,
            timeout=settings.ollama_timeout,
        )
        app.state.ollama_client = llm_client
        logger.info("Ollama client ready")

    # ── Langfuse observability ────────────────────────────────────────────────
    # LangfuseTracer is safe to create even when disabled or unreachable —
    # it checks credentials in __init__ and sets self.client = None if missing.
    # All tracing calls silently no-op when self.client is None.
    langfuse_tracer = LangfuseTracer(settings=settings)
    app.state.langfuse_tracer = langfuse_tracer
    logger.info("Langfuse tracer ready")

    # ── Agentic RAG service ───────────────────────────────────────────────────
    # The graph is compiled once here (validates edges, builds routing tables).
    # Each request gets a fresh Context injected at ainvoke() time — the
    # compiled graph itself is stateless and shared across all requests.
    agentic_rag_service = AgenticRAGService(
        opensearch_client=hybrid_os_client,
        ollama_client=llm_client,
        embeddings_client=jina_client,
        langfuse_tracer=langfuse_tracer,
        graph_config=GraphConfig(),
    )
    app.state.agentic_rag_service = agentic_rag_service
    logger.info("Agentic RAG service ready")

    # ── Redis cache ───────────────────────────────────────────────────────────
    # Redis connection is optional — if Redis is down, the cache is simply
    # skipped and requests proceed normally. We wrap init in try/except so a
    # missing Redis container doesn't prevent the API from starting.
    try:
        redis_client = redis.Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            password=settings.redis.password or None,
            decode_responses=True,
        )
        redis_client.ping()  # fail fast if Redis is unreachable
        app.state.cache_client = CacheClient(redis_client, settings.redis)
        logger.info(f"Redis cache ready ({settings.redis.host}:{settings.redis.port})")
    except Exception as e:
        logger.warning(f"Redis unavailable — caching disabled: {e}")
        app.state.cache_client = None

    logger.info("RAG Assistant API ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await app.state.jina_client.close()
    logger.info("Jina client connection pool closed")
    app.state.langfuse_tracer.shutdown()
    logger.info("Langfuse flushed and shut down")
    logger.info("Shutting down complete")


app = FastAPI(
    title="RAG Assistant",
    description="Production RAG system for intelligent document retrieval and Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(hybrid_search_router)
app.include_router(ask_router)
app.include_router(agentic_ask_router)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Comprehensive health check: tests PostgreSQL, OpenSearch, and Ollama.

    Returns per-service status so operators can see exactly which dependency
    is down without digging through logs. Overall status is "degraded" if any
    single service is unhealthy.
    """
    settings = get_settings()
    services: dict[str, ServiceStatus] = {}
    overall = "ok"

    # PostgreSQL
    try:
        session = get_session()
        session.execute(text("SELECT 1"))
        session.close()
        services["database"] = ServiceStatus(status="healthy", message="Connected successfully")
    except Exception as e:
        services["database"] = ServiceStatus(status="unhealthy", message=str(e))
        overall = "degraded"

    # OpenSearch
    try:
        os_client = OpenSearchClient(host=settings.opensearch.host)
        if os_client.health_check():
            stats = os_client.get_index_stats()
            services["opensearch"] = ServiceStatus(
                status="healthy",
                message=f"Index '{stats.get('index_name', 'unknown')}' — {stats.get('document_count', 0)} documents",
            )
        else:
            services["opensearch"] = ServiceStatus(status="unhealthy", message="Cluster not healthy")
            overall = "degraded"
    except Exception as e:
        services["opensearch"] = ServiceStatus(status="unhealthy", message=str(e))
        overall = "degraded"

    # LLM provider (OpenAI or Ollama)
    if settings.openai.enabled and settings.openai.api_key:
        services["llm"] = ServiceStatus(
            status="healthy",
            message=f"OpenAI API (model: {settings.openai.model})",
        )
    else:
        try:
            ollama = OllamaClient(base_url=settings.ollama_host, timeout=settings.ollama_timeout)
            ollama_health = await ollama.health_check()
            services["llm"] = ServiceStatus(
                status=ollama_health["status"],
                message=ollama_health["message"],
            )
            if ollama_health["status"] != "healthy":
                overall = "degraded"
        except Exception as e:
            services["llm"] = ServiceStatus(status="unhealthy", message=str(e))
            overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
