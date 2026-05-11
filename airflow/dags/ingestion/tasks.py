"""
Airflow task functions for the document ingestion pipeline.

WHY a separate module instead of defining callables inside the DAG file?
────────────────────────────────────────────────────────────────────────
1. TESTABILITY: Functions defined inside a DAG file are hard to unit-test
   because you have to import the whole DAG (which triggers Airflow DAG
   registration and requires the Airflow environment). Functions in a plain
   module can be imported and called directly in pytest.

2. READABILITY: A DAG file should describe STRUCTURE (task names, dependencies,
   schedule). Task business logic mixed in makes the file long and hard to
   navigate.

3. REUSE: The same task function can be used in multiple DAGs (e.g. a backfill
   DAG and a daily DAG) by importing from this module.

HOW AIRFLOW FINDS THIS MODULE:
Airflow adds the `dags/` directory to Python's sys.path. This file lives at
`dags/ingestion/tasks.py`, so it's importable as `ingestion.tasks` from the
DAG file — the same way production uses `arxiv_ingestion.tasks`.

PYTHONPATH NOTE:
The compose.yml sets PYTHONPATH=/opt/airflow/src (so `from config import ...`
works). But we need `from src.config import ...` style imports (because src/
has an __init__.py and our modules use that prefix). We add /opt/airflow to
sys.path at module load time to make both styles work.
"""

import logging
import sys
from datetime import datetime

# Add /opt/airflow so Python can resolve "from src.X import Y"
# (/opt/airflow/src is mounted there, and src/ has __init__.py)
# This mirrors what the production tasks.py does.
sys.path.insert(0, "/opt/airflow")

from sqlalchemy import text

from src.config import get_settings
from src.services.database import get_session, create_tables

# NOTE: IngestionOrchestrator, OpenSearchClient, and sync_all_documents are
# intentionally NOT imported here at module level. Importing them triggers
# Docling's ML model loading which takes 2-4 minutes. Airflow re-parses every
# DAG file every 30 seconds — a 4-minute import kills the DAG processor.
# Instead, each function imports only what it needs right before using it.


def _get_execution_date(context: dict) -> str:
    """
    Extract the execution date string (YYYY-MM-DD) from the Airflow task context.

    WHY this helper exists:
    Airflow 3.0 removed the "ds" context key that existed in Airflow 2.x.
    In Airflow 3.0 the equivalent is "logical_date" (a pendulum.DateTime object).
    For manually triggered runs, even logical_date can be None (no schedule interval).

    This helper tries each source in order and falls back to today's date so
    the task never crashes on a missing key regardless of Airflow version or
    how the run was triggered.
    """
    # Airflow 2.x key
    if context.get("ds"):
        return context["ds"]
    # Airflow 3.x key (pendulum.DateTime object)
    logical_date = context.get("logical_date")
    if logical_date:
        return logical_date.strftime("%Y-%m-%d")
    # Manual trigger with no logical date — use today
    return datetime.now().strftime("%Y-%m-%d")


logger = logging.getLogger(__name__)


def _get_opensearch_client():
    """
    Helper: create an OpenSearchClient using settings from config.
    Import is deferred inside the function to avoid triggering Docling
    model loading at DAG parse time.
    """
    from src.services.opensearch.client import OpenSearchClient
    settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host)


# ─── Task 1: setup_environment ────────────────────────────────────────────────

def setup_environment() -> dict:
    """
    Verify that all external dependencies are reachable BEFORE the pipeline runs.

    WHY a dedicated setup task?
    Without this, a database or OpenSearch failure surfaces midway through the
    pipeline — after you've already spent several minutes fetching and parsing
    papers. Checking upfront means:
      - Failures are immediately visible in the Airflow UI on task 1
      - You don't waste compute resources on steps that will fail anyway
      - Operators know exactly what's down (DB? OpenSearch? Both?)

    This pattern is called "fail fast": surface problems as early as possible
    so the cost of failure is low and the diagnosis is clear.
    """
    logger.info("=== Task 1/5: setup_environment ===")

    # 1. Database — create tables if they don't exist, then verify connectivity.
    # create_tables() is idempotent (uses CREATE TABLE IF NOT EXISTS).
    create_tables()
    session = get_session()
    try:
        session.execute(text("SELECT 1"))
        logger.info("✓ PostgreSQL connection OK")
    finally:
        session.close()

    # 2. OpenSearch — health_check() returns True for green OR yellow status.
    # A single-node dev cluster is always "yellow" (no replicas) — that's fine.
    os_client = _get_opensearch_client()
    if os_client.health_check():
        logger.info("✓ OpenSearch connection OK")

        # create_papers_index(force=False) is idempotent — no-op if index exists.
        # We call it here so the index is guaranteed to exist before the
        # sync task (task 3) tries to write to it.
        created = os_client.create_papers_index(force=False)
        if created:
            logger.info("✓ OpenSearch index created (first run)")
        else:
            logger.info("✓ OpenSearch index already exists")
    else:
        # We log a WARNING rather than raising an exception here.
        # OpenSearch being slow to start shouldn't abort the entire DAG run —
        # the indexing task (task 3) will perform its own health check and
        # skip gracefully if OpenSearch is still down by then.
        logger.warning("⚠ OpenSearch health check failed — indexing may be skipped this run")

    return {"status": "success", "message": "Environment setup complete"}


# ─── Task 2: fetch_daily_papers ───────────────────────────────────────────────

def fetch_daily_papers(**context) -> dict:
    """
    Fetch yesterday's arXiv papers and store them in PostgreSQL.

    CRITICAL DESIGN DECISION: this task does NOT index to OpenSearch.
    Indexing is handled exclusively by task 3. This separation means:
      - If OpenSearch is down, re-running task 3 picks up where task 2 left off
        WITHOUT re-fetching from arXiv (saves API calls and parse time)
      - Each task has a single responsibility (easier to debug failures)
      - Airflow can retry task 3 independently if it fails

    ABOUT **context AND context["ds"]:
    Airflow injects runtime metadata into task functions via **context.
    "ds" is the "data interval start" — the logical execution date in YYYY-MM-DD
    format. It represents the start of the time window this DAG run covers.

    Example: a DAG scheduled for 06:00 UTC on 2025-01-15 will have ds="2025-01-14"
    (the previous day) in most Airflow versions. This is by design: the 06:00 run
    processes yesterday's data.

    WHY use context["ds"] instead of datetime.now()?
    If an Airflow DAG run is re-triggered for a past date (called "backfill"),
    datetime.now() would fetch TODAY's papers instead of that past date's papers.
    context["ds"] always gives the logically correct date for the run.
    """
    execution_date = _get_execution_date(context)
    logger.info(f"=== Task 2/5: fetch_daily_papers (execution_date={execution_date}) ===")

    # Lazy import: IngestionOrchestrator pulls in PDFParser which loads Docling.
    # Docling takes 2-4 min to load its ML models — doing this at module level
    # would kill the DAG processor. Importing here means the cost is paid only
    # when this task actually executes, not every time Airflow parses the DAG.
    from src.services.ingestion.orchestrator import IngestionOrchestrator

    # IngestionOrchestrator.run() coordinates:
    #   1. Fetch paper metadata from arXiv API
    #   2. Download PDFs to local disk
    #   3. Parse PDFs with Docling
    #   4. Upsert documents into PostgreSQL
    orchestrator = IngestionOrchestrator()
    results = orchestrator.run(days_back=1)

    if results.get("fetched", 0) == 0:
        logger.warning(
            f"No papers fetched for {execution_date} — "
            "arXiv may have no new papers (weekend or holiday?)"
        )

    logger.info(
        f"Fetch complete — fetched: {results['fetched']}, "
        f"stored: {results['stored']}, parsed: {results['parsed']}, "
        f"failed: {results['failed']}"
    )

    # XCom (cross-communication): push the stats dict to Airflow's metadata DB.
    # Downstream tasks pull this with:
    #     context["task_instance"].xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")
    #
    # XCom values are serialized as JSON, so they must be JSON-serializable.
    # Our stats dict is plain ints and strings — that's fine.
    # CAUTION: XCom is designed for small metadata (< 1MB). Never push full
    # documents or large lists through XCom — use a shared database or S3 instead.
    context["task_instance"].xcom_push(key="fetch_results", value=results)
    return results


# ─── Task 3: sync_papers_to_opensearch ───────────────────────────────────────

def sync_papers_to_opensearch(**context) -> dict:
    """
    Index the papers stored by task 2 from PostgreSQL into OpenSearch.

    This task is the dedicated OpenSearch write task. It is the ONLY task that
    writes to OpenSearch (task 2 writes only to PostgreSQL). This clean
    separation makes the DAG easy to reason about:
      - PostgreSQL = source of truth (always has the data)
      - OpenSearch = search index (eventually consistent replica)

    FLOW:
    1. Pull fetch results from XCom (to know how many papers to expect)
    2. Health-check OpenSearch (skip gracefully if it's down)
    3. Call sync_all_documents() which reads from PostgreSQL and syncs
    4. Get post-indexing stats from OpenSearch
    5. Push indexing results to XCom for the report task
    """
    logger.info("=== Task 3/5: sync_papers_to_opensearch ===")

    # XCom pull: read what task 2 (fetch_daily_papers) produced.
    fetch_results = context["task_instance"].xcom_pull(
        task_ids="fetch_daily_papers", key="fetch_results"
    )

    if not fetch_results:
        logger.warning("No fetch_results in XCom — was fetch_daily_papers skipped or failed?")
        result = {"status": "skipped", "indexed": 0, "failed": 0, "total_in_index": "unknown"}
        context["task_instance"].xcom_push(key="index_results", value=result)
        return result

    papers_stored = fetch_results.get("stored", 0)
    if papers_stored == 0:
        logger.info("fetch_daily_papers stored 0 papers — nothing to index this run")
        result = {"status": "skipped", "indexed": 0, "failed": 0, "total_in_index": 0}
        context["task_instance"].xcom_push(key="index_results", value=result)
        return result

    logger.info(f"fetch_daily_papers stored {papers_stored} papers — beginning indexing")

    # Check OpenSearch health before attempting to write.
    # If it's down, we return a "failed" status rather than raising an exception —
    # this lets the report task (task 4) still run and record what happened.
    os_client = _get_opensearch_client()
    if not os_client.health_check():
        logger.error("OpenSearch not healthy — skipping indexing this run")
        result = {
            "status": "failed",
            "indexed": 0,
            "failed": papers_stored,
            "total_in_index": "unknown",
        }
        context["task_instance"].xcom_push(key="index_results", value=result)
        return result

    # Lazy import for the same reason as IngestionOrchestrator above.
    from src.services.search_loaders.paper_loader import sync_all_documents

    # sync_all_documents() reads ALL successfully-parsed documents from
    # PostgreSQL and calls OpenSearchClient.bulk_upsert_papers() on them.
    # Returns: {"total": int, "indexed": int, "failed": int, "skipped": int}
    indexing_stats = sync_all_documents()

    # Fetch post-indexing total for the report (e.g. "4,532 total in index")
    try:
        index_stats = os_client.get_index_stats()
        total_in_index = index_stats.get("document_count", "unknown")
    except Exception as exc:
        logger.warning(f"Could not fetch index stats after indexing: {exc}")
        total_in_index = "unknown"

    # Chunk indexing: split papers into overlapping chunks, embed with Jina,
    # and upsert into the arxiv-papers-chunks KNN index for vector search.
    # This runs after paper sync so both BM25 and KNN indices are populated.
    chunks_indexed = 0
    chunks_failed = 0
    try:
        import asyncio
        import concurrent.futures
        from src.services.search_loaders.chunk_loader import ChunkLoader
        from src.services.search_loaders.text_chunker import TextChunker
        from src.services.embeddings.jina_client import JinaEmbeddingsClient
        from src.services.opensearch.client import OpenSearchClient
        from src.models.document import Document

        settings = get_settings()
        def process_paper_in_thread(paper_dict):
            # httpx.AsyncClient is bound to the event loop it was created in.
            # Creating ChunkLoader inside the thread gives it a fresh event loop,
            # avoiding "attached to a different loop" errors.
            async def inner():
                chunk_loader = ChunkLoader(
                    chunker=TextChunker(),
                    embeddings_client=JinaEmbeddingsClient(api_key=settings.jina_api_key),
                    opensearch_client=OpenSearchClient(),
                )
                return await chunk_loader.process_paper(paper_dict)
            return asyncio.run(inner())

        session = get_session()
        try:
            docs = session.query(Document).filter(
                Document.pdf_parsed == "success",
                Document.full_text.isnot(None),
            ).all()
            for doc in docs:
                paper_dict = {
                    "arxiv_id": doc.id,
                    "id": doc.id,
                    "title": doc.title or "",
                    "abstract": doc.abstract or "",
                    "full_text": doc.full_text,
                    "sections": doc.sections,
                    "authors": doc.authors or [],
                }
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(process_paper_in_thread, paper_dict).result()
                    chunks_indexed += 1
                except Exception as exc:
                    logger.warning(f"Chunk indexing failed for {doc.id}: {exc}")
                    chunks_failed += 1
        finally:
            session.close()
        logger.info(f"Chunk indexing complete — papers chunked: {chunks_indexed}, failed: {chunks_failed}")
    except Exception as exc:
        logger.error(f"Chunk indexing step failed: {exc}")

    result = {
        "status": "completed",
        "indexed": indexing_stats["indexed"],
        "failed": indexing_stats.get("failed", 0),
        "skipped": indexing_stats.get("skipped", 0),
        "total_in_index": total_in_index,
        "chunks_indexed": chunks_indexed,
        "chunks_failed": chunks_failed,
    }

    logger.info(
        f"Indexing complete — indexed: {result['indexed']}, "
        f"failed: {result['failed']}, total in index: {total_in_index}, "
        f"chunks: {chunks_indexed}"
    )

    context["task_instance"].xcom_push(key="index_results", value=result)
    return result


# ─── Task 4: generate_daily_report ────────────────────────────────────────────

def generate_daily_report(**context) -> dict:
    """
    Aggregate stats from all pipeline tasks and log a structured daily report.

    WHY a dedicated report task?
    ────────────────────────────
    1. OBSERVABILITY: Having a consistent daily log entry makes it easy to
       grep Airflow logs for trends: "how many papers have we been indexing
       per day this month?"

    2. RUNS AFTER FAILURES: The report task runs even if task 3 failed
       (Airflow still executes downstream tasks with trigger_rule="all_done"
       if configured). This gives you a complete record of what happened.

    3. EXTENSIBILITY: In a real production system, this task would push the
       report dict to a monitoring system (Datadog, PagerDuty, Slack webhook).
       Right now it just logs — but the structure is already correct for that.

    Data source: XCom pulls from task 2 and task 3.
    """
    logger.info("=== Task 4/5: generate_daily_report ===")

    # Pull stats from upstream tasks. Use `or {}` so the code below doesn't
    # crash with AttributeError if a task was skipped and returned None.
    fetch_results = (
        context["task_instance"].xcom_pull(
            task_ids="fetch_daily_papers", key="fetch_results"
        ) or {}
    )
    index_results = (
        context["task_instance"].xcom_pull(
            task_ids="sync_papers_to_opensearch", key="index_results"
        ) or {}
    )

    report = {
        "date": _get_execution_date(context),
        "generated_at": datetime.now().isoformat(),
        "pipeline": {
            "papers_fetched": fetch_results.get("fetched", 0),
            "papers_parsed": fetch_results.get("parsed", 0),
            "papers_stored": fetch_results.get("stored", 0),
            "fetch_failures": fetch_results.get("failed", 0),
        },
        "opensearch": {
            "papers_indexed": index_results.get("indexed", 0),
            "indexing_failures": index_results.get("failed", 0),
            "papers_skipped": index_results.get("skipped", 0),
            "total_in_index": index_results.get("total_in_index", "unknown"),
            "status": index_results.get("status", "unknown"),
        },
    }

    # Log in a format easy to grep: every line starts with a consistent prefix
    logger.info("=" * 60)
    logger.info("DAILY ARXIV INGESTION REPORT")
    logger.info(f"  Date             : {report['date']}")
    logger.info(f"  Generated at     : {report['generated_at']}")
    logger.info("  --- Fetch ---")
    logger.info(f"  Papers fetched   : {report['pipeline']['papers_fetched']}")
    logger.info(f"  Papers parsed    : {report['pipeline']['papers_parsed']}")
    logger.info(f"  Papers stored    : {report['pipeline']['papers_stored']}")
    logger.info(f"  Fetch failures   : {report['pipeline']['fetch_failures']}")
    logger.info("  --- OpenSearch ---")
    logger.info(f"  Papers indexed   : {report['opensearch']['papers_indexed']}")
    logger.info(f"  Indexing failures: {report['opensearch']['indexing_failures']}")
    logger.info(f"  Papers skipped   : {report['opensearch']['papers_skipped']}")
    logger.info(f"  Total in index   : {report['opensearch']['total_in_index']}")
    logger.info(f"  OS status        : {report['opensearch']['status']}")
    logger.info("=" * 60)

    return report
