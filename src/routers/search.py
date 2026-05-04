from fastapi import APIRouter, HTTPException
from src.schemas.search import SearchRequest, SearchResponse
from src.services.opensearch.indexing_service import search_bm25
import logging

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
    """Check if OpenSearch index exists and has documents."""
    from src.services.opensearch.indexing_service import get_opensearch_client, INDEX_NAME
    client = get_opensearch_client()
    
    try:
        stats = client.indices.stats(index=INDEX_NAME)
        doc_count = stats["_all"]["total"]["docs"]["count"]
        return {"status": "healthy", "indexed_documents": doc_count}
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}