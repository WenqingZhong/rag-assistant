import logging
from opensearchpy import OpenSearch, NotFoundError
from src.config import get_settings
from src.services.database import get_session
from src.models.document import Document
from src.services.opensearch.index_config import INDEX_SETTINGS, INDEX_NAME

logger = logging.getLogger(__name__)


def get_opensearch_client() -> OpenSearch:
    """
    Create and return an OpenSearch client.
    
    Reads the host from config (localhost:9200 locally,
    opensearch:9200 inside Docker network).
    """
    settings = get_settings()
    host = settings.opensearch.host.replace("http://", "").replace("https://", "")
    
    return OpenSearch(
        hosts=[{"host": host.split(":")[0], "port": int(host.split(":")[1])}],
        http_compress=True,   # compress requests for speed
        use_ssl=False,
        verify_certs=False,
    )


def create_index_if_not_exists():
    """
    Create the OpenSearch index with our mapping if it doesn't exist.
    Safe to call multiple times — idempotent.
    """
    client = get_opensearch_client()
    
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
        logger.info(f"Created OpenSearch index: {INDEX_NAME}")
    else:
        logger.info(f"OpenSearch index already exists: {INDEX_NAME}")


def index_document(doc: Document) -> bool:
    """
    Index a single document into OpenSearch.
    
    Only indexes documents that were successfully parsed — no point
    searching documents with no full_text content.
    
    Returns True if indexed, False if skipped.
    """
    if doc.pdf_parsed != "success" or not doc.full_text:
        logger.debug(f"Skipping document {doc.id} — not parsed successfully")
        return False

    client = get_opensearch_client()

    # Build the document body for OpenSearch.
    # We don't store the full_text in its entirety for the index body —
    # we truncate it to respect the max_text_size config.
    settings = get_settings()
    full_text = doc.full_text[:settings.opensearch.max_text_size] if doc.full_text else ""

    body = {
        "id": doc.id,
        "source": doc.source,
        "title": doc.title,
        "abstract": doc.abstract or "",
        "full_text": full_text,
        "authors": doc.authors or [],
        "published_date": doc.published_date.isoformat() if doc.published_date else None,
        "pdf_parsed": doc.pdf_parsed,
        # embedding is left empty — populated in Week 4
    }

    # index() is OpenSearch's upsert — creates or updates by document ID.
    # Using doc.id as the OpenSearch document ID means re-indexing
    # the same paper never creates duplicates.
    client.index(
        index=INDEX_NAME,
        id=doc.id,
        body=body,
        refresh=True,   # make document immediately searchable after indexing
                        # (default is async — slight delay before it appears in results)
    )
    logger.info(f"Indexed document: {doc.id} — {doc.title[:50]}")
    return True


def index_all_documents() -> dict:
    """
    Index all successfully parsed documents from PostgreSQL into OpenSearch.
    Called by the Airflow DAG after ingestion completes.
    
    Returns a stats summary.
    """
    create_index_if_not_exists()
    session = get_session()

    try:
        # Only fetch documents that were successfully parsed
        docs = session.query(Document).filter(
            Document.pdf_parsed == "success"
        ).all()

        stats = {"total": len(docs), "indexed": 0, "skipped": 0}

        for doc in docs:
            success = index_document(doc)
            if success:
                stats["indexed"] += 1
            else:
                stats["skipped"] += 1

        logger.info(f"Indexing complete: {stats}")
        return stats

    finally:
        session.close()


def search_bm25(
    query: str,
    size: int = 10,
    source_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """
    Run a BM25 keyword search across title, abstract, and full_text.
    
    Parameters:
        query: the search string
        size: max number of results to return
        source_filter: filter by source ("arxiv", "legal", etc.)
        date_from: ISO date string, e.g. "2024-01-01"
        date_to: ISO date string, e.g. "2024-12-31"
    
    Returns a dict with total count and list of hits.
    """
    client = get_opensearch_client()

    # multi_match searches across multiple fields at once.
    # "fields" with ^ sets the boost — title^3 means title matches
    # count 3x more toward the score than abstract or full_text matches.
    # This reflects the intuition: if your search term is in the title,
    # that document is almost certainly what you want.
    must_clause = {
        "multi_match": {
            "query": query,
            "fields": [
                "title^3",      # title matches are 3x more valuable
                "abstract^2",   # abstract matches are 2x more valuable
                "full_text",    # body matches count normally
                "authors",      # can search by author name too
            ],
            "type": "best_fields",  # use the best-scoring field for ranking
            "fuzziness": "AUTO",    # tolerate minor typos automatically
        }
    }

    # Filters narrow results without affecting score.
    # "must" affects score. "filter" doesn't — it just includes/excludes.
    filter_clauses = []

    if source_filter:
        filter_clauses.append({"term": {"source": source_filter}})

    if date_from or date_to:
        date_range = {}
        if date_from:
            date_range["gte"] = date_from   # greater than or equal
        if date_to:
            date_range["lte"] = date_to     # less than or equal
        filter_clauses.append({"range": {"published_date": date_range}})

    # Build the full query using OpenSearch Query DSL
    # This is equivalent to SQL:
    # SELECT * FROM documents
    # WHERE (BM25_SCORE(query) > 0)
    #   AND source = source_filter
    #   AND published_date BETWEEN date_from AND date_to
    # ORDER BY BM25_SCORE DESC
    # LIMIT size
    opensearch_query = {
        "query": {
            "bool": {
                "must": [must_clause],
                "filter": filter_clauses,
            }
        },
        "size": size,
        # highlight shows which parts of the document matched the query
        # useful for displaying search snippets in the UI
        "highlight": {
            "fields": {
                "title": {},
                "abstract": {"fragment_size": 200, "number_of_fragments": 1},
            }
        },
        # Only return these fields — don't send full_text back in every result
        # (full_text can be 100k+ chars, way too much for a search result list)
        "_source": ["id", "title", "abstract", "authors", "published_date", "source"],
    }

    try:
        response = client.search(index=INDEX_NAME, body=opensearch_query)
        
        hits = response["hits"]["hits"]
        total = response["hits"]["total"]["value"]

        # Normalize the OpenSearch response into a clean format
        results = []
        for hit in hits:
            result = {
                **hit["_source"],           # spread the document fields
                "score": hit["_score"],     # BM25 relevance score
                "highlight": hit.get("highlight", {}),  # matched snippets
            }
            results.append(result)

        return {"total": total, "results": results, "query": query}

    except NotFoundError:
        # Index doesn't exist yet — return empty results instead of crashing
        logger.warning(f"Index {INDEX_NAME} not found. Run indexing first.")
        return {"total": 0, "results": [], "query": query}