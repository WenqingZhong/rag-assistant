FROM apache/airflow:3.0.0-python3.12

USER root
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY airflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY --chown=airflow:root pyproject.toml /opt/airflow/pyproject.toml

USER airflow

RUN pip install \
    fastapi uvicorn sqlalchemy psycopg2-binary \
    requests httpx python-dateutil docling

ENTRYPOINT ["/entrypoint.sh"]