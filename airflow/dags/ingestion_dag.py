# This file lives in airflow/dags/ — Airflow automatically scans
# this folder and registers any DAG it finds. 

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def run_ingestion():
    """
    The actual work this DAG does — this is the function Airflow calls.

    WHY the sys.path manipulation?
    Inside the Airflow container, Python doesn't automatically know
    where src/ code lives. sys.path is Python's list of directories
    to search when you do an import.
    We add "/opt/airflow/src" (where we mounted src/ folder
    in compose.yml) so that "from src.services..." imports work.
    Without this, Python would throw ModuleNotFoundError.
    """
    import sys
    # Insert at position 0 = highest priority in the search path.
    # This ensures our src/ is found before any system packages
    # that might have conflicting names.
    sys.path.insert(0, "/opt/airflow/src")
    from src.services.ingestion.orchestrator import IngestionOrchestrator

    orchestrator = IngestionOrchestrator()
    stats = orchestrator.run(days_back=1)

    print(f"Ingestion complete: {stats}")
    return stats


with DAG(
    # Unique identifier for this DAG — shows up in the Airflow UI.
    dag_id="document_ingestion",

    description="Daily document ingestion pipeline",

    # Cron expression: "0 6 * * 1-5"
    # ┌─ minute (0)
    # │  ┌─ hour (6 = 6 AM UTC)
    # │  │  ┌─ day of month (* = any)
    # │  │  │  ┌─ month (* = any)
    # │  │  │  │  ┌─ day of week (1-5 = Monday to Friday)
    # 0  6  *  *  1-5
    # Runs at 6 AM UTC every weekday.
    # arXiv publishes new papers around 5 AM UTC
    schedule="0 6 * * 1-5",

    # start_date tells Airflow "this DAG exists from this date onwards."
    # It does NOT mean "run immediately on this date."
    # Combined with catchup=False, it just sets a reference point.
    start_date=datetime(2025, 1, 1),

    # catchup=False is IMPORTANT.
    # If catchup=True (the default), Airflow would try to run the DAG
    # for every scheduled slot between start_date and now.
    # That means from Jan 2025 to today = hundreds of backfill runs all at once.
    # catchup=False says "ignore missed runs, only run going forward."
    catchup=False,

    # Prevent two runs of this DAG from happening simultaneously.
    # If today's run is still going when tomorrow's is scheduled,
    # tomorrow's waits instead of starting a second parallel run.
    # Important because our pipeline downloads and writes files —
    # two concurrent runs could conflict on the same file paths.
    max_active_runs=1,

    # Default settings applied to every task in this DAG.
    # Individual tasks can override these if needed.
    default_args={
        # If a task fails, retry it 2 times before marking it as failed.
        "retries": 2,

        # Wait 30 minutes between retries.
        "retry_delay": timedelta(minutes=30),
    },

) as dag:

    # Define the single task in this DAG.
    # PythonOperator = "run this Python function as a task."
    # Airflow has many other operators: BashOperator, HttpOperator,
    # EmailOperator, etc. PythonOperator is the most flexible.
    ingest = PythonOperator(
        # Unique name for this task within the DAG.
        task_id="ingest_documents",
        python_callable=run_ingestion,
    )

    # If you had multiple tasks you'd define dependencies here with >>
    # For example:
    #   fetch_task >> download_task >> parse_task >> index_task
    # The >> operator means "this task must complete before the next one starts."