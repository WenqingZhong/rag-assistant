from typing import Dict, List

from pydantic import BaseModel


class JinaEmbeddingRequest(BaseModel):
    """
    Request body for the Jina AI embeddings API.

    WHY two different task values?
    Jina's jina-embeddings-v3 is a *task-aware* model — it produces
    different vector representations depending on the declared task:

      task="retrieval.passage"  →  used when indexing document chunks.
                                   The model optimises for being retrieved.

      task="retrieval.query"    →  used when embedding a user's search query.
                                   The model optimises for finding passages.

    Using mismatched tasks (e.g. passage task for the query) degrades
    retrieval quality because the two vector spaces aren't aligned.
    Always use "retrieval.passage" for index-time and "retrieval.query"
    for search-time.

    WHY dimensions=1024?
    Jina v3 supports 32–1024 dimensions. 1024 gives the best retrieval
    quality at the cost of more storage (1024 floats × 4 bytes = 4 KB per
    chunk). For a few thousand papers this is negligible.
    """
    model: str = "jina-embeddings-v3"
    task: str = "retrieval.passage"   # override to "retrieval.query" for search queries
    dimensions: int = 1024
    late_chunking: bool = False       # Jina feature: context-aware chunking — not used here
    embedding_type: str = "float"     # float32 vectors
    input: List[str]                  # the texts to embed


class JinaEmbeddingResponse(BaseModel):
    """
    Response from the Jina AI embeddings API.

    The `data` list has one entry per input text, each containing:
        {"embedding": [float, ...], "index": int, "object": "embedding"}

    We access data[i]["embedding"] to get the vector for input[i].
    """
    model: str
    object: str = "list"
    usage: Dict[str, int]   # {"prompt_tokens": N, "total_tokens": N}
    data: List[Dict]        # list of {"embedding": List[float], "index": int}
