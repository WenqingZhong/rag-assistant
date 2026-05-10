import logging

from src.config import get_settings
from src.services.database import get_session
from src.models.document import Document
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

    In a high-traffic API we'd cache this in FastAPI's app.state and inject
    it via dependency injection. For our Airflow + low-RPS API workloads,
    per-call creation is perfectly fine.
    """
    settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host)


# ─── Schema management ────────────────────────────────────────────────────────

def ensure_papers_index():
    """
    Create the papers OpenSearch index if it doesn't already exist.
    Safe to call repeatedly — idempotent (no-op if index exists).
    """
    _get_client().create_papers_index(force=False)


# ─── PostgreSQL → OpenSearch loaders ─────────────────────────────────────────

def sync_document(doc: Document) -> bool:
    """
    Load a single Document from PostgreSQL into OpenSearch.

    Only loads documents with pdf_parsed == "success" AND non-empty full_text.
    There is no value in loading a document with no text content: the user
    couldn't find it through full-text search anyway. Metadata-only documents
    (no PDF) are still stored in PostgreSQL for reference but skipped here.

    Returns True if the document was written, False if skipped or errored.
    """
    if doc.pdf_parsed != "success" or not doc.full_text:
        logger.debug(f"Skipping '{doc.id}' — pdf_parsed={doc.pdf_parsed}, has_text={bool(doc.full_text)}")
        return False

    paper_data = {
        "id": doc.id,
        "source": doc.source,
        "title": doc.title,
        "abstract": doc.abstract or "",
        "full_text": doc.full_text,
        "authors": doc.authors or [],
        "published_date": doc.published_date.isoformat() if doc.published_date else None,
        "pdf_parsed": doc.pdf_parsed,
    }

    return _get_client().upsert_paper(paper_data)


def sync_all_documents() -> dict:
    """
    Load all successfully parsed documents from PostgreSQL into OpenSearch.

    Called by the Airflow task after the fetch task has stored papers in PostgreSQL.

    Design note: we fetch ALL successful documents, not just today's.
    This is intentional — if a previous day's load failed, running again will
    pick up those documents too. OpenSearch's upsert semantics mean re-loading
    the same document is safe (just an update, no duplicates).

    Returns a stats dict for XCom communication and logging:
        {"total": int, "indexed": int, "failed": int, "skipped": int}
    """
    ensure_papers_index()
    session = get_session()

    try:
        docs = session.query(Document).filter(Document.pdf_parsed == "success").all()

        papers_to_write = []
        skipped = 0

        for doc in docs:
            if not doc.full_text:
                skipped += 1
                continue
            papers_to_write.append({
                "id": doc.id,
                "source": doc.source,
                "title": doc.title,
                "abstract": doc.abstract or "",
                "full_text": doc.full_text,
                "authors": doc.authors or [],
                "published_date": doc.published_date.isoformat() if doc.published_date else None,
                "pdf_parsed": doc.pdf_parsed,
            })

        result = _get_client().bulk_upsert_papers(papers_to_write)

        stats = {
            "total": len(docs),
            "indexed": result["success"],
            "failed": result["failed"],
            "skipped": skipped,
        }
        logger.info(f"sync_all_documents complete: {stats}")
        return stats

    finally:
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
    Run a BM25 keyword search across all loaded documents.

    Returns: {"total": int, "results": [...], "query": str}
    """
    return _get_client().search_papers(
        query=query,
        size=size,
        source_filter=source_filter,
        date_from=date_from,
        date_to=date_to,
    )
