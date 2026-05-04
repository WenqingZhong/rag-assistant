import logging

from fastapi import APIRouter, HTTPException

from src.config import get_settings
from src.schemas.api.search import SearchRequest, SearchResponse
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.indexing_service import search_bm25

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/")
def search(request: SearchRequest) -> SearchResponse:
    """
    BM25 keyword search across all indexed documents.

    POST body example:
    {
        "query": "transformer attention mechanism",
        "size": 5,
        "source": "arxiv",
        "date_from": "2024-01-01"
    }

    The call chain:
        search_bm25() → OpenSearchClient.search_papers() → PaperQueryBuilder.build()
    """
    try:
        results = search_bm25(
            query=request.query,
            size=request.size,
            source_filter=request.source,
            date_from=request.date_from,
            date_to=request.date_to,
        )
        return SearchResponse(**results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/health")
def search_health():
    """
    Check OpenSearch availability and index statistics.

    Returns document count, index size, and cluster health status.
    Useful for operators to confirm the index is populated and healthy
    before trusting search results.
    """
    settings = get_settings()
    client = OpenSearchClient(host=settings.opensearch.host)

    # health_check() tests the cluster, not just connectivity.
    # Returns False if status is "red" (primary shards unassigned).
    if not client.health_check():
        return {"status": "unavailable", "error": "OpenSearch cluster is not healthy"}

    stats = client.get_index_stats()

    if "error" in stats:
        return {"status": "unavailable", "error": stats["error"]}

    return {
        "status": "healthy",
        "indexed_documents": stats["document_count"],
        "index_size_bytes": stats["size_in_bytes"],
        "cluster_health": stats["health"],
    }
