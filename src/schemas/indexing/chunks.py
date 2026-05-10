from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """
    Positional and structural metadata for a single text chunk.

    WHY track character offsets and word counts?
    When debugging retrieval quality ("why did this chunk match?"), you need
    to be able to locate the chunk precisely in the original document.
    start_char / end_char let you highlight the exact span in the source text.

    WHY track overlap?
    The chunker uses sliding windows — consecutive chunks share overlap_size
    words. Knowing how much a chunk overlaps with its neighbors helps
    downstream consumers avoid presenting near-duplicate content to users.
    """
    chunk_index: int                    # position of this chunk within the paper (0-based)
    start_char: int                     # character offset in the original text where this chunk starts
    end_char: int                       # character offset where this chunk ends
    word_count: int                     # number of words in the chunk text
    overlap_with_previous: int          # words shared with the preceding chunk
    overlap_with_next: int              # words shared with the following chunk
    section_title: Optional[str] = None # section heading this chunk came from (None for word-based fallback)


class TextChunk(BaseModel):
    """
    A single chunk of text ready for embedding and indexing.

    This is the central data structure in the week 4 pipeline:
        TextChunker.chunk_paper() → List[TextChunk]
        JinaEmbeddingsClient.embed_passages([chunk.text]) → List[List[float]]
        ChunkLoader.process_paper() → OpenSearch

    WHY include arxiv_id and paper_id on every chunk?
    Each chunk gets its own OpenSearch document. When a search returns a chunk,
    you need to trace it back to the paper in PostgreSQL (paper_id) and display
    the source citation (arxiv_id). Denormalizing these fields onto every chunk
    avoids a database round-trip per search result.
    """
    text: str               # the chunk text sent to Jina for embedding
    metadata: ChunkMetadata
    arxiv_id: str           # e.g. "2301.07041" — used in citations and chunk deletion
    paper_id: str           # PostgreSQL row ID — used to JOIN back to full paper data
