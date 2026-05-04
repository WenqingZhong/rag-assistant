from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


def run_ingestion():
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from src.services.ingestion.orchestrator import IngestionOrchestrator
    orchestrator = IngestionOrchestrator()
    stats = orchestrator.run(days_back=1)
    print(f"Ingestion complete: {stats}")
    return stats


def run_indexing():
    """
    Index all successfully parsed documents into OpenSearch.
    Runs after ingestion so new documents are immediately searchable.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/src")
    from src.services.opensearch.indexing_service import index_all_documents
    stats = index_all_documents()
    print(f"Indexing complete: {stats}")
    return stats


with DAG(
    dag_id="document_ingestion",
    description="Daily document ingestion and indexing pipeline",
    schedule="0 6 * * 1-5",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=30),
    },
) as dag:

    ingest = PythonOperator(
        task_id="ingest_documents",
        python_callable=run_ingestion,
    )

    index = PythonOperator(
        task_id="index_documents",
        python_callable=run_indexing,
    )

    # >> means "ingest must finish before index starts"
    ingest >> index