FROM apache/airflow:3.0.0-python3.12

USER root
RUN apt-get update && apt-get install -y \
    curl \
    libxcb1 \
    libx11-6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY airflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY --chown=airflow:root pyproject.toml /opt/airflow/pyproject.toml

USER airflow

RUN pip install \
    fastapi uvicorn sqlalchemy psycopg2-binary \
    requests httpx python-dateutil docling opensearch-py

ENTRYPOINT ["/entrypoint.sh"]