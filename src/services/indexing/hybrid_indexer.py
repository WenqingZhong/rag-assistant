import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class HybridIndexingService:
    """
    Orchestrates the full pipeline: paper text → chunks → embeddings → OpenSearch.

    WHY a separate service class instead of putting this logic in OpenSearchClient?
    OpenSearchClient is responsible for ONE thing: talking to OpenSearch.
    HybridIndexingService coordinates THREE services (chunker, Jina, OpenSearch)
    and owns the business logic of the pipeline. Mixing that into OpenSearchClient
    would make it impossible to swap out the chunker or embeddings provider without
    touching OpenSearch code.

    This is dependency injection: the three collaborators are passed in at
    construction time rather than instantiated here. Benefits:
      - Tests can inject a mock Jina client without making real API calls.
      - The caller (main.py / Airflow task) controls the lifecycle of each client.
      - You can reuse the same OpenSearchClient for both paper indexing and
        chunk indexing without creating two separate connections.

    WHY async?
    embed_passages() is async (it makes an HTTP call to Jina's API). Any method
    that awaits it must also be async. index_paper() and index_papers_batch()
    are both async so they can be awaited from FastAPI route handlers or Airflow
    async tasks without blocking the event loop.
    """

    def __init__(
        self,
        chunker: TextChunker,
        embeddings_client: JinaEmbeddingsClient,
        opensearch_client: OpenSearchClient,
    ):
        self.chunker = chunker
        self.embeddings_client = embeddings_client
        self.opensearch_client = opensearch_client
        logger.info("HybridIndexingService initialised")

    # ── Public API ────────────────────────────────────────────────────────────

    async def index_paper(self, paper_data: Dict) -> Dict[str, int]:
        """
        Run the full 4-step pipeline for one paper.

        Step 1 — Chunk:
            TextChunker.chunk_paper() tries section-based chunking first,
            falls back to sliding-window word chunking. Returns List[TextChunk].

        Step 2 — Embed:
            JinaEmbeddingsClient.embed_passages() sends all chunk texts to
            Jina's API in batches (50 at a time) and returns one 1024-dim
            vector per chunk. This is the only async step.

        Step 3 — Prepare:
            Pack each (chunk, embedding) pair into the flat dict format that
            matches ARXIV_PAPERS_CHUNKS_MAPPING. Denormalize paper metadata
            (title, authors, abstract) onto every chunk document so search
            results can display a useful card without a JOIN.

        Step 4 — Index:
            OpenSearchClient.bulk_index_chunks() sends all prepared documents
            to OpenSearch in a single bulk request.

        :param paper_data: Dict from PostgreSQL — must have arxiv_id, title,
                           abstract, full_text. sections and authors are optional.
        :returns: {
            "chunks_created":       int,  # how many chunks TextChunker produced
            "chunks_indexed":       int,  # how many OpenSearch accepted
            "embeddings_generated": int,  # should equal chunks_created
            "errors":               int,  # failed chunks
          }
        """
        arxiv_id = paper_data.get("arxiv_id")
        paper_id = str(paper_data.get("id", ""))

        if not arxiv_id:
            logger.error("Cannot index paper: missing arxiv_id")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        try:
            # ── Step 1: Chunk ──────────────────────────────────────────────
            chunks = self.chunker.chunk_paper(
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                # "raw_text" is the week-4 field name; "full_text" is ours.
                # Support both so this works with either DB schema.
                full_text=paper_data.get("full_text", paper_data.get("raw_text", "")),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                sections=paper_data.get("sections"),
            )

            if not chunks:
                logger.warning(f"No chunks produced for paper {arxiv_id} — nothing to index")
                return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 0}

            logger.info(f"Step 1 complete: {len(chunks)} chunks for {arxiv_id}")

            # ── Step 2: Embed ──────────────────────────────────────────────
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embeddings_client.embed_passages(
                texts=chunk_texts,
                batch_size=50,
                # WHY 50? Jina supports up to 2048 texts per request but
                # large batches with long texts can hit token limits.
                # 50 chunks × ~600 words = ~30k tokens per batch — safe margin.
            )

            if len(embeddings) != len(chunks):
                # This should never happen unless Jina drops items from a batch.
                # Fail the whole paper rather than indexing mis-aligned chunks.
                logger.error(
                    f"Embedding count mismatch for {arxiv_id}: "
                    f"{len(embeddings)} embeddings for {len(chunks)} chunks"
                )
                return {
                    "chunks_created": len(chunks),
                    "chunks_indexed": 0,
                    "embeddings_generated": len(embeddings),
                    "errors": 1,
                }

            logger.info(f"Step 2 complete: {len(embeddings)} embeddings for {arxiv_id}")

            # ── Step 3: Prepare ────────────────────────────────────────────
            chunks_with_embeddings = []
            authors = self._normalise_authors(paper_data.get("authors"))
            published_date = self._normalise_date(paper_data.get("published_date"))

            for chunk, embedding in zip(chunks, embeddings):
                # chunk_id is deterministic: re-indexing the same paper produces
                # the same IDs, which means bulk_index_chunks acts as an upsert.
                chunk_id = f"{arxiv_id}_chunk_{chunk.metadata.chunk_index}"

                chunk_doc = {
                    # Identity
                    "chunk_id": chunk_id,
                    "arxiv_id": chunk.arxiv_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    # Content
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "section_title": chunk.metadata.section_title,
                    "embedding_model": "jina-embeddings-v3",
                    # Denormalized paper metadata — stored on every chunk so
                    # search results carry enough context to render a result
                    # card (title, authors, abstract) without a DB round-trip.
                    "title": paper_data.get("title", ""),
                    "authors": authors,
                    "abstract": paper_data.get("abstract", ""),
                    "categories": paper_data.get("categories", []),
                    "published_date": published_date,
                }
                chunks_with_embeddings.append({"chunk_data": chunk_doc, "embedding": embedding})

            # ── Step 4: Index ──────────────────────────────────────────────
            results = self.opensearch_client.bulk_index_chunks(chunks_with_embeddings)

            logger.info(
                f"Step 4 complete for {arxiv_id}: "
                f"{results['success']} indexed, {results['failed']} failed"
            )
            return {
                "chunks_created": len(chunks),
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }

        except Exception as e:
            logger.error(f"Error indexing paper {arxiv_id}: {e}", exc_info=True)
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

    async def index_papers_batch(
        self,
        papers: List[Dict],
        replace_existing: bool = False,
    ) -> Dict[str, int]:
        """
        Index a list of papers sequentially, aggregating stats.

        WHY sequential, not concurrent?
        Each paper calls Jina's API (rate-limited) and OpenSearch's bulk API.
        Running 15 papers concurrently would fire 15 Jina requests simultaneously —
        likely hitting Jina's rate limit and causing retries. Sequential is safer
        and still fast enough for our daily batch (typically 5-15 papers).

        :param papers:           List of paper dicts from PostgreSQL.
        :param replace_existing: If True, delete existing chunks for each paper
                                 before re-indexing. Use when re-parsing PDFs.
        :returns: Aggregated stats across all papers.
        """
        totals = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for paper in papers:
            arxiv_id = paper.get("arxiv_id")

            if replace_existing and arxiv_id:
                deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
                if deleted:
                    logger.info(f"Deleted {deleted} existing chunks for {arxiv_id}")

            stats = await self.index_paper(paper)

            totals["papers_processed"] += 1
            totals["total_chunks_created"] += stats["chunks_created"]
            totals["total_chunks_indexed"] += stats["chunks_indexed"]
            totals["total_embeddings_generated"] += stats["embeddings_generated"]
            totals["total_errors"] += stats["errors"]

        logger.info(
            f"Batch complete: {totals['papers_processed']} papers, "
            f"{totals['total_chunks_indexed']}/{totals['total_chunks_created']} chunks indexed"
        )
        return totals

    async def reindex_paper(self, arxiv_id: str, paper_data: Dict) -> Dict[str, int]:
        """
        Delete all existing chunks for a paper, then re-index from scratch.

        Use when the PDF was re-parsed (better text extraction) or when the
        chunking/embedding configuration changed.
        """
        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
        logger.info(f"Deleted {deleted} chunks before reindexing {arxiv_id}")
        return await self.index_paper(paper_data)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalise_authors(self, authors: Optional[object]) -> str:
        """
        Normalise the authors field to a plain string.

        PostgreSQL stores authors as a TEXT[] array, which SQLAlchemy returns
        as a Python list. OpenSearch's text field accepts both a list and a
        string for indexing, but storing as a joined string is simpler and
        avoids ambiguity in the result payload.

        Example:
            ["Vaswani, A.", "Shazeer, N."] → "Vaswani, A., Shazeer, N."
            "Vaswani, A."                  → "Vaswani, A."   (already a string)
            None                           → ""
        """
        if isinstance(authors, list):
            return ", ".join(authors)
        return str(authors) if authors else ""

    def _normalise_date(self, published_date: Optional[object]) -> Optional[str]:
        """
        Normalise published_date to an ISO 8601 string for OpenSearch.

        OpenSearch's date field requires "YYYY-MM-DD" or full ISO 8601.
        SQLAlchemy may return a Python date, datetime, or already-formatted string.

        Example:
            date(2024, 1, 15)            → "2024-01-15"
            datetime(2024, 1, 15, 0, 0)  → "2024-01-15"
            "2024-01-15"                 → "2024-01-15"
            None                         → None
        """
        if published_date is None:
            return None
        if isinstance(published_date, (date, datetime)):
            return published_date.isoformat()[:10]  # keep only YYYY-MM-DD
        return str(published_date)
