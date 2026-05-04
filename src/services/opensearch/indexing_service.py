import logging

from src.config import get_settings
from src.services.database import get_session
from src.models.document import Document
from src.services.opensearch.index_config import INDEX_NAME  # kept for import compatibility
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def _get_client() -> OpenSearchClient:
    """
    Create a fresh OpenSearchClient for this call.

    WHY not a module-level singleton?
    ──────────────────────────────────
    A module-level client is instantiated when the module is first imported.
    Imports happen at process startup — before OpenSearch is necessarily
    ready (especially in Docker where services race to become healthy).

    Creating per-call is a small overhead (a Python object + dict),
    NOT a new TCP connection — opensearchpy pools connections internally.
    So this is safe and avoids startup-time connection errors.

    In a high-traffic API you'd cache this in FastAPI's app.state and inject
    it via dependency injection. For our Airflow + low-RPS API workloads,
    per-call creation is perfectly fine.
    """
    settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host)


# ─── Index management ─────────────────────────────────────────────────────────

def create_index_if_not_exists():
    """
    Create the OpenSearch index with our mapping if it doesn't already exist.
    Safe to call repeatedly — idempotent (no-op if index exists).
    """
    _get_client().create_index(force=False)


# ─── Document indexing ────────────────────────────────────────────────────────

def index_document(doc: Document) -> bool:
    """
    Index a single Document (from the PostgreSQL documents table) into OpenSearch.

    Only indexes documents with pdf_parsed == "success" AND non-empty full_text.
    There is no value in indexing a document with no text content: the user
    couldn't find it through full-text search anyway. Metadata-only documents
    (no PDF) are still stored in PostgreSQL for reference but skipped here.

    Returns True if the document was indexed, False if skipped or errored.
    """
    if doc.pdf_parsed != "success" or not doc.full_text:
        logger.debug(f"Skipping '{doc.id}' — pdf_parsed={doc.pdf_parsed}, has_text={bool(doc.full_text)}")
        return False

    paper_data = {
        "id": doc.id,
        "source": doc.source,
        "title": doc.title,
        "abstract": doc.abstract or "",
        # full_text and authors are passed as-is; OpenSearchClient handles
        # truncation (full_text) and list→string conversion (authors).
        "full_text": doc.full_text,
        "authors": doc.authors or [],
        "published_date": doc.published_date.isoformat() if doc.published_date else None,
        "pdf_parsed": doc.pdf_parsed,
    }

    return _get_client().index_paper(paper_data)


def index_all_documents() -> dict:
    """
    Index all successfully parsed documents from PostgreSQL into OpenSearch.

    Called by the Airflow indexing task (task 3 of 5) after the fetch task
    has stored papers in PostgreSQL.

    Design note: we fetch ALL successful documents, not just today's.
    This is intentional — if a previous day's indexing run failed, running
    again will pick up those documents too. OpenSearch's upsert semantics
    mean re-indexing the same document is safe (just an update).

    Returns a stats dict for XCom communication and logging:
        {"total": int, "indexed": int, "failed": int, "skipped": int}
    """
    create_index_if_not_exists()
    session = get_session()

    try:
        docs = session.query(Document).filter(Document.pdf_parsed == "success").all()

        # Build the list of dicts to pass to bulk_index_papers().
        # We separate "skipped" (no full_text despite success status) here
        # before handing off to the client — the client doesn't know about
        # the Document model's meaning of pdf_parsed.
        papers_to_index = []
        skipped = 0

        for doc in docs:
            if not doc.full_text:
                skipped += 1
                continue
            papers_to_index.append({
                "id": doc.id,
                "source": doc.source,
                "title": doc.title,
                "abstract": doc.abstract or "",
                "full_text": doc.full_text,
                "authors": doc.authors or [],
                "published_date": doc.published_date.isoformat() if doc.published_date else None,
                "pdf_parsed": doc.pdf_parsed,
            })

        # bulk_index_papers() calls index_paper() per document and collects
        # success/failure counts. Returns {"success": int, "failed": int}.
        result = _get_client().bulk_index_papers(papers_to_index)

        stats = {
            "total": len(docs),
            "indexed": result["success"],
            "failed": result["failed"],
            "skipped": skipped,
        }
        logger.info(f"index_all_documents complete: {stats}")
        return stats

    finally:
        # Always close the session even if an exception was raised.
        # Leaving sessions open causes connection pool exhaustion over time.
        session.close()


# ─── Search ───────────────────────────────────────────────────────────────────

def search_bm25(
    query: str,
    size: int = 10,
    source_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """
    Run a BM25 keyword search across all indexed documents.

    This function's SIGNATURE is unchanged from before — the router calls
    it with the same arguments it always did. Internally, it now delegates
    to OpenSearchClient.search_papers(), which delegates query construction
    to PaperQueryBuilder.

    The three-layer chain:
        search_bm25()           → business-level interface (this module)
            ↓
        OpenSearchClient        → service layer (manages connection & response)
            ↓
        PaperQueryBuilder       → query construction (builds the DSL dict)
            ↓
        opensearchpy.OpenSearch → raw HTTP to OpenSearch

    Parameters:
        query:         search string
        size:          max results (default 10)
        source_filter: filter by source field (e.g. "arxiv")
        date_from:     ISO date lower bound (e.g. "2024-01-01")
        date_to:       ISO date upper bound (e.g. "2024-12-31")

    Returns: {"total": int, "results": [...], "query": str}
    """
    return _get_client().search_papers(
        query=query,
        size=size,
        source_filter=source_filter,
        date_from=date_from,
        date_to=date_to,
    )
