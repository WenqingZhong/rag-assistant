import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.schemas.api.search import HybridSearchRequest, HybridSearchResponse, SearchHit
from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hybrid-search", tags=["hybrid-search"])


# ── Dependency providers ──────────────────────────────────────────────────────
# These read the long-lived client instances from app.state (set in main.py).
# WHY app.state instead of instantiating clients here?
# JinaEmbeddingsClient holds a persistent httpx connection pool that is reused
# across requests. Creating a new client per request would open and close a TLS
# connection on every search call — ~100ms of unnecessary latency each time.
# OpenSearchClient similarly holds a connection pool.
# Storing them in app.state means one instance is created at startup and shared
# across all requests for the lifetime of the process.

def get_opensearch_client(request: Request) -> OpenSearchClient:
    return request.app.state.hybrid_opensearch_client


def get_jina_client(request: Request) -> JinaEmbeddingsClient:
    return request.app.state.jina_client


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    opensearch_client: OpenSearchClient = Depends(get_opensearch_client),
    jina_client: JinaEmbeddingsClient = Depends(get_jina_client),
) -> HybridSearchResponse:
    """
    Hybrid search: BM25 keyword matching + KNN vector similarity, merged by RRF.

    Results are chunk-level (600-word passages), not whole papers.
    Each hit includes the matching chunk text, its section, and the paper metadata.

    Request body:
        {
            "query": "transformer attention mechanism",
            "size": 10,
            "categories": ["cs.AI"],   // optional
            "use_hybrid": true,        // false = BM25 only (no Jina call)
            "min_score": 0.0           // drop results below this RRF score
        }

    Search mode decision:
        use_hybrid=True  AND Jina call succeeds  →  "hybrid" (BM25 + KNN + RRF)
        use_hybrid=True  AND Jina call fails      →  "bm25"   (graceful fallback)
        use_hybrid=False                          →  "bm25"   (explicit override)

    WHY graceful fallback instead of raising 503 when Jina fails?
    A keyword search result is far more useful than an error page. If Jina's API
    is down, the user still gets relevant papers — they just lose the semantic
    ranking. The response includes search_mode="bm25" so the UI can optionally
    show a degraded-mode badge.
    """
    if not opensearch_client.health_check():
        raise HTTPException(status_code=503, detail="Search service unavailable")

    # ── Step 1: Optionally embed the query ────────────────────────────────────
    query_embedding = None
    search_mode = "bm25"

    if request.use_hybrid:
        try:
            query_embedding = await jina_client.embed_query(request.query)
            search_mode = "hybrid"
            logger.info(f"Query embedded for hybrid search: '{request.query[:60]}'")
        except Exception as e:
            # Jina is unavailable — fall back to BM25 rather than failing the request.
            logger.warning(f"Jina embedding failed, falling back to BM25: {e}")
            query_embedding = None
            search_mode = "bm25"

    # ── Step 2: Execute search ────────────────────────────────────────────────
    logger.info(
        f"Hybrid search [{search_mode}]: '{request.query}' "
        f"size={request.size} categories={request.categories}"
    )

    raw_results = opensearch_client.search_unified(
        query=request.query,
        query_embedding=query_embedding,
        size=request.size,
        from_=request.from_,
        categories=request.categories,
        use_hybrid=request.use_hybrid,
        min_score=request.min_score,
    )

    # ── Step 3: Map raw OpenSearch hits → SearchHit objects ───────────────────
    # OpenSearch returns raw dicts; Pydantic validates and types them here.
    # Fields not present in the hit (e.g. section_title for word-based chunks)
    # default to None via the SearchHit schema.
    hits = [
        SearchHit(
            arxiv_id=hit.get("arxiv_id", ""),
            title=hit.get("title", ""),
            authors=hit.get("authors"),
            abstract=hit.get("abstract"),
            published_date=str(hit["published_date"]) if hit.get("published_date") else None,
            score=hit.get("score", 0.0),
            chunk_text=hit.get("chunk_text"),
            chunk_id=hit.get("chunk_id"),
            section_title=hit.get("section_title"),
            highlights=hit.get("highlights"),
        )
        for hit in raw_results.get("hits", [])
    ]

    logger.info(f"Hybrid search complete: {len(hits)} hits returned (mode={search_mode})")

    return HybridSearchResponse(
        query=request.query,
        total=raw_results.get("total", 0),
        hits=hits,
        size=request.size,
        search_mode=search_mode,
    )
