from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.routers.documents import router as documents_router
import logging

from src.services.database import create_tables

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG Assistant API...")
    create_tables()  # ← always ensure tables exist on startup
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Assistant",
    description="Production RAG system for intelligent document retrieval and Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)


@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}