import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from src.config import get_settings
from src.routers.documents import router as documents_router
from src.routers.search import router as search_router
from src.schemas.api.health import HealthResponse, ServiceStatus
from src.services.database import create_tables, get_session
from src.services.ollama import OllamaClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG Assistant API...")
    create_tables()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Assistant",
    description="Production RAG system for intelligent document retrieval and Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(search_router)


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

    # Ollama
    try:
        ollama = OllamaClient(base_url=settings.ollama_host, timeout=settings.ollama_timeout)
        ollama_health = await ollama.health_check()
        services["ollama"] = ServiceStatus(
            status=ollama_health["status"],
            message=ollama_health["message"],
        )
        if ollama_health["status"] != "healthy":
            overall = "degraded"
    except Exception as e:
        services["ollama"] = ServiceStatus(status="unhealthy", message=str(e))
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
