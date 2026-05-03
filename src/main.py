from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    logger.info("Starting up arXiv Paper Curator API...")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="arXiv Paper Curator",
    description="Production RAG system for academic papers",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}