# RAG Assistant — Master Command Reference

## Docker Operations

### Start / Stop
```bash
docker compose up -d                    # start all containers (no rebuild)
docker compose up --build -d            # start + rebuild changed images
docker compose up --build --force-recreate -d  # force full rebuild
docker compose down                     # stop all containers (keeps data)
docker compose down -v                  # ⚠️ DANGER: stop + delete all volumes (wipes data)
docker compose restart api              # restart one service (no rebuild)
```

### Status & Logs
```bash
docker compose ps                       # show all containers + status
docker compose logs -f api              # tail logs for api service
docker compose logs -f airflow          # tail logs for airflow
docker compose logs airflow | tail -30  # last 30 lines of airflow logs
docker compose logs airflow | grep -i "password\|login\|admin"  # find airflow password
```

### Debugging
```bash
docker compose build airflow --no-cache --progress=plain 2>&1 | tail -50  # verbose build output
docker exec rag-api cat /app/src/main.py          # read file inside container
docker exec rag-api grep -n "get_session" /app/src/services/database.py   # search file inside container
docker ps -a | grep airflow             # show all containers including stopped
docker rm rag-airflow                   # remove a stopped container
docker container prune                  # remove ALL stopped containers
docker volume ls | grep postgres        # check if volume exists
```

---

## Health Checks

### All Services
```bash
# FastAPI
curl http://localhost:8000/api/v1/health

# OpenSearch cluster
curl http://localhost:9200/_cluster/health

# OpenSearch index stats
curl http://localhost:9200/documents/_stats

# PostgreSQL
docker exec rag-postgres psql -U rag_user -d rag_db -c "SELECT version();"

# Ollama models
docker exec rag-ollama ollama list
```

### Browser UIs
```
http://localhost:8000/docs    FastAPI Swagger UI (all endpoints)
http://localhost:5601         OpenSearch Dashboards
http://localhost:8080         Airflow UI (login: admin / see logs for password)
```

---

## Database (PostgreSQL)

### Check Documents
```bash
# Quick count
docker exec rag-postgres psql -U rag_user -d rag_db \
  -c "SELECT COUNT(*) FROM documents;"

# List documents with parse status
docker exec rag-postgres psql -U rag_user -d rag_db \
  -c "SELECT id, title, pdf_parsed FROM documents;"

# Detailed view
docker exec rag-postgres psql -U rag_user -d rag_db \
  -c "SELECT title, pdf_parsed, length(full_text) FROM documents LIMIT 5;"

# List all tables
docker exec rag-postgres psql -U rag_user -d rag_db -c "\dt"
```

### Via Python
```bash
docker exec rag-api python -c "
from src.services.database import get_session, create_tables
from src.models.document import Document

create_tables()  # ensure tables exist
session = get_session()
total = session.query(Document).count()
parsed = session.query(Document).filter(Document.pdf_parsed == 'success').count()
failed = session.query(Document).filter(Document.pdf_parsed == 'failed').count()
pending = session.query(Document).filter(Document.pdf_parsed == 'pending').count()
print(f'Total: {total}, Parsed: {parsed}, Failed: {failed}, Pending: {pending}')

doc = session.query(Document).first()
if doc:
    print(f'Sample title: {doc.title}')
    print(f'Has full_text: {doc.full_text is not None}')
    print(f'full_text length: {len(doc.full_text) if doc.full_text else 0}')
session.close()
"
```

### GUI Connection (TablePlus / DBeaver)
```
Type:     PostgreSQL
Host:     localhost
Port:     5432
Database: rag_db
User:     rag_user
Password: rag_password
```

---

## Ingestion Pipeline

### Run Manually
```bash
# Run pipeline for last N days
docker exec rag-api python -c "
from src.services.ingestion.orchestrator import IngestionOrchestrator
o = IngestionOrchestrator()
stats = o.run(days_back=2)
print(stats)
"

# Larger backfill (if data was lost)
docker exec rag-api python -c "
from src.services.ingestion.orchestrator import IngestionOrchestrator
o = IngestionOrchestrator()
stats = o.run(days_back=5)
print(stats)
"
```

### Expected Output
```python
{'fetched': 15, 'stored': 15, 'parsed': 10, 'failed': 5}
# fetched = papers from arXiv API
# stored  = saved to PostgreSQL (even if PDF parsing failed)
# parsed  = successfully parsed by Docling
# failed  = Docling couldn't parse (normal for complex PDFs)
```

---

## OpenSearch

### Index Management
```bash
# Check index exists
curl http://localhost:9200/documents

# Check how many documents are indexed
curl http://localhost:9200/documents/_count

# Delete index (use when mapping changes)
curl -X DELETE http://localhost:9200/documents

# Check index mapping
curl http://localhost:9200/documents/_mapping
```

### Run Indexing Manually
```bash
docker exec rag-api python -c "
from src.services.opensearch.indexing_service import index_all_documents
stats = index_all_documents()
print(stats)
"
```

### Search via API
```bash
# Basic BM25 search
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "neural network transformer", "size": 3}'

# Search with filters
curl -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RAG retrieval",
    "size": 5,
    "source": "arxiv",
    "date_from": "2024-01-01"
  }'

# Search health check
curl http://localhost:8000/api/v1/search/health
```

### Search Directly Against OpenSearch (bypass API)
```bash
curl -X POST http://localhost:9200/documents/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "multi_match": {
        "query": "RAG",
        "fields": ["title^3", "abstract^2", "full_text"]
      }
    },
    "size": 3
  }'
```

---

## Documents API
```bash
# List all documents (paginated)
curl http://localhost:8000/api/v1/documents/

# List with pagination
curl "http://localhost:8000/api/v1/documents/?limit=5&offset=0"

# Get specific document
curl http://localhost:8000/api/v1/documents/2301.07041
```

---

## Airflow

### Access UI
```
URL:      http://localhost:8080
Username: admin
Password: check logs → docker compose logs airflow | grep -i "password"
```

### Trigger DAG Manually
```bash
# Always check for queued/running runs before triggering to avoid stacking up runs
docker exec rag-airflow airflow dags list-runs document_ingestion

# Trigger only if nothing is queued or running
docker exec rag-airflow airflow dags trigger document_ingestion

# Or use the Airflow UI at http://localhost:8080
```

### Check DAG Status
```bash
# List all runs with their state
docker exec rag-airflow airflow dags list-runs document_ingestion

# Check which tasks are running and their state for a specific run
docker exec rag-postgres psql -U rag_user -d rag_db \
  -c "SELECT task_id, state, start_date FROM task_instance
      WHERE dag_id = 'document_ingestion'
      ORDER BY start_date DESC LIMIT 10;"

# Watch live task output
docker compose logs -f airflow | grep -E "setup_env|fetch_daily|index_papers|generate_daily|cleanup|ERROR|failed"
```

### Cancel Stacked Queued Runs
```bash
# If you accidentally triggered multiple runs, cancel the queued ones
docker exec rag-airflow airflow dags pause document_ingestion

docker exec rag-postgres psql -U rag_user -d rag_db \
  -c "UPDATE dag_run SET state = 'failed'
      WHERE dag_id = 'document_ingestion'
      AND state = 'queued';"

docker exec rag-airflow airflow dags unpause document_ingestion
```

### Airflow Login
```bash
# Password is regenerated on every container restart
docker compose logs airflow | grep "Password for user"
# Username is always: admin
```

### Unpause a DAG (new DAGs start paused by default)
```bash
docker exec rag-airflow airflow dags unpause document_ingestion
```

---

## Ollama (LLM)

```bash
# List downloaded models
docker exec rag-ollama ollama list

# Pull a model
docker exec rag-ollama ollama pull llama3.2:1b

# Test LLM response (takes 10-30s on CPU)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:1b",
  "prompt": "Say hello in one sentence.",
  "stream": false
}'
```

---

## Common Fixes

### Container name conflict
```bash
docker rm <container_name>
# or remove all stopped:
docker container prune
```

### Changes not reflected after rebuild
```bash
docker compose down
docker compose up --build --force-recreate -d
```

### OpenSearch index mapping changed
```bash
curl -X DELETE http://localhost:9200/documents
# then re-run indexing
```

### Data missing after restart
```bash
# Check if tables exist
docker exec rag-postgres psql -U rag_user -d rag_db -c "\dt"

# Recreate tables + re-run pipeline
docker exec rag-api python -c "
from src.services.database import create_tables
create_tables()
"
# then run ingestion pipeline
```

### arXiv 429 rate limit error
```
Wait 5-10 minutes, then retry.
Set ARXIV__RATE_LIMIT_DELAY=5.0 in .env to reduce frequency.
```

### Airflow webserver not loading
```bash
# Check logs for errors
docker compose logs airflow | tail -30

# Airflow 3.0 — correct commands
airflow db migrate        # initialize database
airflow standalone        # start everything (NOT airflow webserver)
```
