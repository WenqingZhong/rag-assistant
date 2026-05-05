from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Existing BM25 search schemas (used by /api/v1/search) ────────────────────
# These are kept intact — changing them would break the existing search endpoint.

class SearchRequest(BaseModel):
    """
    Request body for the BM25 paper search endpoint (/api/v1/search).
    Searches the arxiv-papers index (full papers, not chunks).
    """
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    size: int = Field(default=10, ge=1, le=50, description="Number of results")
    source: Optional[str] = Field(default=None, description="Filter by source (e.g. 'arxiv')")
    date_from: Optional[str] = Field(default=None, description="Filter from date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(default=None, description="Filter to date (YYYY-MM-DD)")


class SearchResult(BaseModel):
    """A single result from the BM25 paper search (full-paper level)."""
    id: str
    title: str
    abstract: Optional[str] = None
    authors: list = []
    published_date: Optional[str] = None
    source: str
    score: float
    highlight: dict = {}


class SearchResponse(BaseModel):
    """Response from the BM25 paper search endpoint."""
    total: int
    results: list[SearchResult]
    query: str


# ── Hybrid search schemas (used by /api/v1/hybrid-search) ────────────────────
# These operate at the CHUNK level, not the paper level.
# Each hit is a matching passage (600 words) with its embedding-based score,
# plus the paper metadata (title, authors, etc.) denormalized onto it.

class HybridSearchRequest(BaseModel):
    """
    Request body for the hybrid search endpoint (/api/v1/hybrid-search).

    Supports three modes controlled by use_hybrid:
        use_hybrid=True  (default): BM25 on chunk_text + KNN on embedding, merged by RRF.
                                    Requires a Jina API key to embed the query.
        use_hybrid=False:           BM25 only on the chunks index (no Jina call).
                                    Useful when Jina is unavailable or for debugging.

    WHY min_score?
    RRF scores have no fixed scale — they depend on how many candidates each
    sub-query returned. min_score lets callers discard weakly-ranked results
    (e.g. a chunk that only appeared at rank 40 in BM25 and rank 45 in KNN).
    The default of 0.0 returns everything; 0.01 is a reasonable noise filter.
    """
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    size: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    from_: int = Field(default=0, ge=0, alias="from", description="Offset for pagination")
    categories: Optional[List[str]] = Field(
        default=None, description="Filter by arXiv categories (e.g. ['cs.AI', 'cs.LG'])"
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid search (BM25 + vector). False = BM25 only.",
    )
    min_score: float = Field(
        default=0.0, ge=0.0, description="Minimum RRF score threshold — drop weaker results"
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "query": "transformer attention mechanism",
                "size": 10,
                "categories": ["cs.AI", "cs.LG"],
                "use_hybrid": True,
                "min_score": 0.0,
            }
        },
    }


class SearchHit(BaseModel):
    """
    A single result from the hybrid search — one matching chunk, not a whole paper.

    WHY chunk-level results instead of paper-level?
    The query matched a specific 600-word passage. Returning the whole paper
    would bury the relevant section in thousands of words. The LLM that reads
    these results only needs the matching passage, not the full text.

    Paper metadata (title, authors, abstract) is denormalized here — it was
    stored on every chunk document at index time so we can render a full result
    card without a JOIN back to the papers table.

    chunk_text:   The actual passage that matched — sent directly to the LLM.
    chunk_id:     "<arxiv_id>_chunk_<N>" — deterministic, used for dedup.
    section_title: Which section of the paper this chunk came from
                   (e.g. "Methods (part 2)"). None for word-based chunks.
    highlights:   HTML-marked snippets showing which words matched the query.
                  e.g. {"chunk_text": ["...the <mark>attention</mark> layer..."]}
    score:        RRF combined score (hybrid) or BM25 score (bm25-only mode).
    """
    arxiv_id: str
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    published_date: Optional[str] = None
    score: float
    chunk_text: Optional[str] = Field(None, description="Text content of the matching chunk")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    section_title: Optional[str] = Field(None, description="Section this chunk came from")
    highlights: Optional[Dict] = Field(None, description="Highlighted matching snippets")


class HybridSearchResponse(BaseModel):
    """
    Response from the hybrid search endpoint.

    search_mode tells the caller which path was taken:
        "hybrid" — BM25 + KNN + RRF  (query_embedding was generated and used)
        "bm25"   — BM25 only          (use_hybrid=False or Jina unavailable)

    WHY report search_mode in the response?
    The caller (UI or LLM orchestrator) needs to know whether the results are
    semantically ranked or keyword-ranked. A UI might show a badge "Semantic search"
    vs "Keyword search". An orchestrator might retry with use_hybrid=False if it
    sees search_mode="bm25" unexpectedly.
    """
    query: str
    total: int
    hits: List[SearchHit]
    size: int
    search_mode: str = Field(description="Search mode used: 'hybrid' or 'bm25'")
    error: Optional[str] = None

    model_config = {"populate_by_name": True}
