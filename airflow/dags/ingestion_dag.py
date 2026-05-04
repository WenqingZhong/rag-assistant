"""
Daily arXiv paper ingestion pipeline.

STRUCTURE (5 tasks in sequence):
    setup_environment
        ↓
    fetch_daily_papers        ← fetches from arXiv API, stores in PostgreSQL
        ↓
    index_papers_to_opensearch ← reads from PostgreSQL, writes to OpenSearch
        ↓
    generate_daily_report      ← aggregates stats from tasks 2 & 3 via XCom
        ↓
    cleanup_temp_files         ← removes PDFs older than 30 days from /tmp

WHY 5 tasks instead of 2?
Previously the DAG had two tasks: "ingest" (fetch+store) and "index".
The production pattern splits this further because:

1. setup_environment (new):
   Fail fast. Verify DB and OpenSearch are reachable BEFORE spending minutes
   fetching and parsing papers. A failed dependency is immediately visible on
   task 1, not buried in a stack trace on task 2.

2. fetch_daily_papers (was "ingest_documents"):
   Renamed to be explicit about what it fetches. Now deliberately does NOT
   index to OpenSearch — that responsibility belongs to task 3 alone.

3. index_papers_to_opensearch (was "index_documents"):
   Same name, but now it receives fetch stats from task 2 via XCom and can
   skip gracefully if fetch stored nothing. Pushes indexing stats for task 4.

4. generate_daily_report (new):
   Pulls XCom from both task 2 and task 3 to produce a structured log entry.
   This is the observability layer — one place to see what happened each day.
   In production, this task would push to Slack, Datadog, PagerDuty, etc.

5. cleanup_temp_files (new, BashOperator):
   Removes PDFs older than 30 days from /tmp. PDFs are large (5-30 MB each),
   and after parsing their content is in PostgreSQL — we don't need the file.
   Using a BashOperator for simple shell commands avoids writing Python
   boilerplate for what is essentially one `find -delete` command.

XCOM FLOW:
    fetch_daily_papers  → pushes key="fetch_results"  → pulled by tasks 3 & 4
    index_papers_to_opensearch → pushes key="index_results" → pulled by task 4
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Task functions live in a separate module (ingestion/tasks.py).
# Airflow adds the dags/ directory to sys.path, so this import works as
# if ingestion/ is a top-level package.
from ingestion.tasks import (
    fetch_daily_papers,
    generate_daily_report,
    index_papers_to_opensearch,
    setup_environment,
)

# default_args apply to every task in the DAG unless overridden per-task.
default_args = {
    "owner": "rag-assistant",
    # depends_on_past=False: each DAG run is independent.
    # If True, a run would wait for the PREVIOUS run to succeed before starting.
    # False is correct here — a failed Monday run shouldn't block Tuesday's run.
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    # retries=2: each task will retry up to 2 times before marking as FAILED.
    "retries": 2,
    # retry_delay: wait 30 minutes between retries. This gives transient issues
    # (network blip, OpenSearch restart) time to resolve before retrying.
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    dag_id="document_ingestion",
    description="Daily arXiv paper pipeline: fetch → PostgreSQL → OpenSearch",
    # Start unpaused so manual triggers and scheduled runs work immediately
    # without needing to manually toggle the DAG in the UI each time.
    is_paused_upon_creation=False,
    # schedule: run Monday–Friday at 06:00 UTC.
    # Cron syntax: minute hour day-of-month month day-of-week
    # "1-5" = Monday through Friday
    schedule="0 6 * * 1-5",
    start_date=datetime(2025, 1, 1),
    # catchup=False: if the DAG was paused for a week, don't run 5 missed DAG
    # runs when you re-enable it. Just run once for the current schedule slot.
    catchup=False,
    # max_active_runs=1: prevents two pipeline runs from writing to OpenSearch
    # simultaneously. The ingestion pipeline is not designed for parallelism.
    max_active_runs=1,
    default_args=default_args,
    tags=["arxiv", "ingestion", "opensearch"],
) as dag:

    # ── Task 1: Environment check ────────────────────────────────────────────
    setup_task = PythonOperator(
        task_id="setup_environment",
        python_callable=setup_environment,
        # no **context needed — setup_environment() takes no arguments
    )

    # ── Task 2: Fetch from arXiv, store to PostgreSQL ────────────────────────
    fetch_task = PythonOperator(
        task_id="fetch_daily_papers",
        python_callable=fetch_daily_papers,
        # provide_context=True is implicit in modern Airflow (2.x+) when the
        # callable accepts **context. The context dict includes "ds",
        # "task_instance", "dag_run", etc.
    )

    # ── Task 3: Index from PostgreSQL to OpenSearch ──────────────────────────
    index_task = PythonOperator(
        task_id="index_papers_to_opensearch",
        python_callable=index_papers_to_opensearch,
    )

    # ── Task 4: Generate and log the daily report ────────────────────────────
    report_task = PythonOperator(
        task_id="generate_daily_report",
        python_callable=generate_daily_report,
    )

    # ── Task 5: Remove stale PDFs from /tmp ──────────────────────────────────
    # BashOperator is appropriate here: this is a simple shell command, not
    # Python business logic. No Python wrapper needed.
    #
    # `|| true` at the end prevents the task from failing if find returns
    # a non-zero exit code (e.g. no matching files — that's fine).
    cleanup_task = BashOperator(
        task_id="cleanup_temp_files",
        bash_command="""
        echo "Cleaning up PDFs older than 30 days from /tmp..."
        find /tmp -name "*.pdf" -type f -mtime +30 -delete 2>/dev/null || true
        echo "Cleanup complete"
        """,
    )

    # ── Task dependencies ────────────────────────────────────────────────────
    # The >> operator sets "runs before" relationships.
    # Reading left to right: setup must complete before fetch starts, etc.
    setup_task >> fetch_task >> index_task >> report_task >> cleanup_task
