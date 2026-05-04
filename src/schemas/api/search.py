from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    """
    Request body for the search endpoint.
    Pydantic validates types and provides defaults automatically.
    """
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    size: int = Field(default=10, ge=1, le=50, description="Number of results")
    source: Optional[str] = Field(default=None, description="Filter by source (e.g. 'arxiv')")
    date_from: Optional[str] = Field(default=None, description="Filter from date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(default=None, description="Filter to date (YYYY-MM-DD)")


class SearchResult(BaseModel):
    """A single search result."""
    id: str
    title: str
    abstract: Optional[str] = None
    authors: list = []
    published_date: Optional[str] = None
    source: str
    score: float
    highlight: dict = {}


class SearchResponse(BaseModel):
    """Response from the search endpoint."""
    total: int
    results: list[SearchResult]
    query: str
