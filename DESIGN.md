# RAG Assistant — Design Deep Dive

A reference document covering architecture decisions, tradeoffs, and the reasoning behind each layer of the system. Structured for a 1-hour technical deep dive.

---

## System Overview

A document ingestion and retrieval pipeline for arXiv papers:

```
arXiv API → Airflow (fetch + parse + index) → PostgreSQL + OpenSearch → FastAPI (search)
```

Three independent subsystems:
- **Ingestion** (Airflow): scheduled background pipeline that fetches, parses, and indexes papers
- **Storage** (PostgreSQL): source of truth for document metadata and full text
- **Search** (OpenSearch + FastAPI): BM25 keyword search over indexed documents

---

## 1. Airflow DAG Design

### Structure: 5 tasks in sequence

```
setup_environment
    ↓
fetch_daily_papers          ← arXiv API → PDF download → parse → PostgreSQL
    ↓
index_papers_to_opensearch  ← PostgreSQL → OpenSearch
    ↓
generate_daily_report       ← aggregates XCom stats from tasks 2 & 3
    ↓
cleanup_temp_files          ← BashOperator: removes PDFs older than 30 days
```

### Why split into 5 tasks instead of 1 or 2?

**Fail-fast principle (task 1):** If PostgreSQL or OpenSearch is unreachable, the pipeline should fail immediately on task 1 — not after spending 10 minutes downloading and parsing papers. Each failure maps to a specific task in the Airflow UI, making diagnosis instant.

**Single-responsibility (tasks 2 vs 3):** Task 2 writes only to PostgreSQL. Task 3 reads from PostgreSQL and writes only to OpenSearch. If OpenSearch goes down, you can re-run task 3 alone without re-fetching from arXiv. If task 2 fails mid-batch, task 3 never runs — you don't get a partial index out of sync with the DB.

**Observability (task 4):** A dedicated report task aggregates XCom data from both upstream tasks and logs a structured daily summary. In production this would push to Slack or Datadog. Right now it's logs, but the structure is already correct.

**BashOperator for cleanup (task 5):** `find /tmp -name "*.pdf" -mtime +30 -delete` is one shell command. Writing Python boilerplate to do the same thing adds complexity without value. BashOperator is the right tool here.

### Why a separate `ingestion/tasks.py` module?

Three reasons:
1. **Testability** — functions in the DAG file require the full Airflow environment to import. Functions in a plain module can be unit-tested with `pytest` by importing them directly.
2. **Readability** — the DAG file describes structure (task names, dependencies, schedule). Business logic mixed in makes it long and hard to navigate.
3. **Reuse** — the same functions can be used in a backfill DAG by importing from this module.

### The lazy import pattern (critical for Airflow)

Airflow's DAG processor re-parses every `.py` file in the `dags/` directory every 30 seconds. If `tasks.py` imports Docling (or any heavy ML library) at module level, the DAG processor spends 2-4 minutes loading ML models every 30 seconds → SIGKILL.

**Rule:** Only import at module level what is needed to define the function signatures. Everything else goes inside the function body.

```python
# WRONG — kills the DAG processor
from src.services.ingestion.orchestrator import IngestionOrchestrator

def fetch_daily_papers(**context):
    orchestrator = IngestionOrchestrator()

# CORRECT — import paid only when the task actually runs
def fetch_daily_papers(**context):
    from src.services.ingestion.orchestrator import IngestionOrchestrator
    orchestrator = IngestionOrchestrator()
```

### `.airflowignore`

`dags/.airflowignore` contains `ingestion/` to prevent Airflow from parsing `ingestion/tasks.py` as a DAG file. Without this, Airflow tries to find DAG objects in every `.py` file it can reach, generating noisy errors.

### XCom — cross-task communication

XCom (cross-communication) is Airflow's mechanism for passing small values between tasks. Values are JSON-serialized and stored in Airflow's metadata database (PostgreSQL in this setup).

```
fetch_daily_papers  →  xcom_push(key="fetch_results", value={"fetched": 15, ...})
                              ↓
index_papers_to_opensearch  ←  xcom_pull(task_ids="fetch_daily_papers", key="fetch_results")
                              ↓
generate_daily_report  ←  xcom_pull from both tasks 2 and 3
```

**What XCom is NOT for:** Large data. XCom is designed for metadata (counts, status, IDs). The actual documents live in PostgreSQL. Never push full document content through XCom.

### Airflow 3.0 compatibility

Airflow 3.0 removed `context["ds"]` (the execution date string from 2.x). The replacement is `context["logical_date"]` (a pendulum DateTime object), which can also be `None` for manually triggered runs with no schedule interval.

The `_get_execution_date(context)` helper handles all three cases:
```python
def _get_execution_date(context):
    if context.get("ds"):           # Airflow 2.x
        return context["ds"]
    logical_date = context.get("logical_date")
    if logical_date:                # Airflow 3.x
        return logical_date.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")  # manual trigger, no date
```

---

## 2. OpenSearch Architecture

### Class hierarchy

```
PaperQueryBuilder          — translates search intent into OpenSearch Query DSL
OpenSearchClient           — owns the connection, exposes clean operations
indexing_service.py        — public API unchanged; internally delegates to OpenSearchClient
```

### PaperQueryBuilder — Builder Pattern

The builder separates *what you want* from *how OpenSearch DSL expresses it*. Instead of building a raw dict wherever search is needed, callers pass intent:

```python
PaperQueryBuilder(
    query="RAG retrieval",
    size=5,
    source_filter="arxiv",
    date_from="2024-01-01",
)
```

The builder translates this into a bool query with `must` (BM25 scoring) and `filter` (cached, no score impact) clauses. OpenSearch caches `filter` results separately from scoring — using `filter` for date ranges and source fields is a meaningful performance decision, not just organization.

**`must` vs `filter`:**
- `must`: affects BM25 relevance score. Use for the text query.
- `filter`: binary yes/no, result is cached, does not affect score. Use for date ranges, source type, category.

### Why store authors as a list, not a joined string?

OpenSearch text fields handle arrays natively. Storing `["Alice Wang", "Bob Chen"]` means you can search by individual author name without splitting. Storing `"Alice Wang, Bob Chen"` means the comma becomes part of the indexed token — searching "Wang" might not match depending on the analyzer.

The bug we fixed: `", ".join(doc.authors)` in `index_paper()` converted the list to a string before indexing. When the search returned results, the API tried to validate `"Alice Wang, Bob Chen"` as `list[str]` and raised a `ValidationError`. Fix: remove the join entirely.

### Single index, upsert semantics

`bulk_index_papers()` uses `index` action (not `create`), which means re-indexing a document that already exists updates it in place. This makes the indexing task idempotent — running it twice produces the same result.

### Health check — green OR yellow

A single-node development cluster is always `yellow` (OpenSearch requires replicas for `green` status, and a single node can't replicate to itself). The `health_check()` method returns `True` for both `green` and `yellow`. Requiring `green` would make every dev environment look broken.

---

## 3. PDF Parsing

### Evolution: Docling → pypdfium2

| Approach | Memory | Speed | Outcome |
|----------|--------|-------|---------|
| Docling `StandardPdfPipeline` | ~1-2 GB (layout ML model + TableFormer) | 30s/paper | SIGKILL after 1-2 papers |
| `do_table_structure=False` | ~1 GB (layout model still loads) | ~30s/paper | SIGKILL on first paper |
| Docling `SimplePipeline` | Negligible | Instant | Fails: SimplePipeline doesn't support PDF format |
| **pypdfium2** | Negligible | <1s/paper | **Works** |

### Why not Docling in production?

Docling's `StandardPdfPipeline` loads a layout-analysis ML model (CNN-based document understanding) at inference time, consuming ~1-2 GB. In a Docker container on a laptop sharing RAM with PostgreSQL, OpenSearch, Airflow, and the API, this reliably triggers the OOM killer.

Docling is the right tool on a machine with 16+ GB available to the container. For a dev/laptop environment, `pypdfium2` (Google's PDFium engine as a Python binding) is the practical choice: text extraction in milliseconds, no ML inference, already installed as a transitive dependency of Docling.

### The singleton converter pattern (and why it matters)

Original code re-created `DocumentConverter()` inside `parse()` on every call. Each creation reloaded model weights. On the second PDF, two sets of weights coexisted in RAM simultaneously — the first one not yet GC'd when the second loaded.

Fix: class-level `_converter = None` with lazy initialization. Weights load once on the first call and are reused for all subsequent PDFs within the same process.

With pypdfium2 this is less critical (no model weights), but the pattern remains correct.

### Why `max_pages` matters

A 30-page academic paper might take 30x longer to parse than a 1-page paper. The `max_pages` setting exists to bound per-document parsing time. Previously this setting existed in the config but was never passed to Docling's `convert()`. With pypdfium2, it's enforced in the page loop:

```python
num_pages = min(len(pdf), self.settings.max_pages)
for i in range(num_pages):
    ...
```

---

## 4. Data Model and Storage

### PostgreSQL as source of truth

PostgreSQL holds everything: metadata (title, abstract, authors, categories), parsing status (`pdf_parsed`), and full text. OpenSearch is a derived index — if it were wiped, it could be fully reconstructed from PostgreSQL by running `index_all_documents()`.

This separation is intentional:
- PostgreSQL has ACID guarantees. OpenSearch is eventually consistent.
- Re-indexing is cheap (seconds). Re-fetching from arXiv is expensive (rate limits, bandwidth).
- Task 2 and task 3 are independently retryable because they write to different stores.

### `pdf_parsed` state machine

```
pending → success   (parse succeeded)
pending → failed    (parse threw an exception or returned empty text)
```

`index_all_documents()` filters for `pdf_parsed = 'success'` AND non-empty `full_text`. Documents stay in PostgreSQL regardless of parse status — metadata (title, abstract) is stored even when PDF parsing fails.

### SQLAlchemy 1.4 vs 2.0

Airflow 3.0 ships with SQLAlchemy 1.4. Our `Document` model originally used `DeclarativeBase` (SQLAlchemy 2.0-only). The fix: use `declarative_base()` (function form), which exists in both 1.4 and 2.0.

Forcing `sqlalchemy>=2.0` in the Airflow Dockerfile breaks Airflow itself — Airflow 3.0's internal `TaskInstance` model uses 1.4-style annotations that fail under 2.0's strict mode.

---

## 5. Infrastructure

### Volume mount vs image layer

| | API container | Airflow container |
|---|---|---|
| `src/` location | Baked into image at `/app/src` | Volume-mounted from host at `/opt/airflow/src` |
| To pick up code change | `docker compose up --build -d api` | Immediate — file on disk IS the file in container |
| Why the difference? | API is a deployed service — reproducible image is important | Airflow DAG and task code changes frequently during development |

`data/arxiv_pdfs/` is volume-mounted so PDFs persist across container rebuilds.

### PYTHONPATH and import paths

The Airflow container sets `PYTHONPATH=/opt/airflow/src`, which makes `from config import ...` work. But our code uses `from src.config import ...` (because `src/` has an `__init__.py`). The fix in `tasks.py`:

```python
sys.path.insert(0, "/opt/airflow")  # enables "from src.X import Y"
```

This is done once at module load in `tasks.py` and applies to the whole process.

---

## Deep Dive Questions

**Q: Why does the indexing task check `papers_stored == 0` and skip instead of always calling `index_all_documents()`?**

A: Optimization for the common case — if nothing was fetched this run, there's nothing new to index and the skip is correct. The tradeoff is that documents which were parsed in a previous failed run won't be picked up by this guard. For that case, you'd manually trigger with `index_all_documents()` directly, or change the guard to always index when there are unindexed success documents.

**Q: What happens if OpenSearch goes down while the index task is running?**

A: `bulk_index_papers()` will fail. The task returns a `"failed"` status dict (rather than raising an exception) so the report task (task 4) can still run and record what happened. The documents remain in PostgreSQL with `pdf_parsed='success'` and will be picked up on the next successful run of `index_all_documents()`.

**Q: Why not use a vector database instead of OpenSearch BM25?**

A: BM25 (keyword matching) and vector search (semantic similarity) are complementary. BM25 is exact and explainable — if a user searches "transformer attention mechanism", BM25 will find documents containing those exact words. Vector search finds semantically similar documents even without keyword overlap. A production RAG system would use hybrid retrieval (BM25 + vector rerank). This project implements BM25 first because it's infrastructure-simple and the quality bar for search is already high when users know what they're looking for.

**Q: Why does the arXiv fetcher fetch `days_back=1` — won't weekends have no papers?**

A: Yes — arXiv doesn't publish on weekends. The DAG is also only scheduled Monday–Friday (`0 6 * * 1-5`). On Monday it fetches Friday's papers. Weekend gaps are expected and handled gracefully: `fetch_daily_papers` logs a warning when 0 papers are fetched, and downstream tasks skip cleanly.

**Q: How would you scale this pipeline to 10x the paper volume?**

A: Three bottlenecks to address: (1) PDF parsing is the slowest step — switch back to Docling on a machine with sufficient RAM, or run multiple Airflow workers with `CeleryExecutor`. (2) arXiv rate limits — `rate_limit_delay=3.0` seconds between requests means ~300 papers/15 minutes max; increase `max_concurrent_downloads` carefully. (3) OpenSearch bulk indexing — current batch size is one call per run; for 10x volume, shard the indexing across multiple `bulk_index_papers()` calls with bounded batch sizes.

**Q: What does `is_paused_upon_creation=False` do and why does it matter?**

A: New DAGs in Airflow start paused by default — no runs execute until an operator manually toggles them in the UI. `is_paused_upon_creation=False` means the DAG is immediately active when Airflow first parses it. Without this, every deploy requires a manual unpause step that's easy to forget.

**Q: Why is XCom stored in the metadata database and not a message queue?**

A: XCom is designed for small coordination data (task stats, IDs, status), not for passing large payloads. Storing it in the metadata database keeps the system simple — no separate message broker needed. The constraint is size: XCom should be under ~1MB per value. Passing the full document list between tasks would be the wrong use — that's what PostgreSQL is for.

**Q: What is the index named `documents` vs `arxiv-papers`?**

A: A naming inconsistency from early development. The `documents` index was created manually before the `OpenSearchClient` class existed. `OpenSearchClient` uses the index name from config (`opensearch.index_name`, default `arxiv-papers`). The old `documents` index is stale and can be deleted with `curl -X DELETE http://localhost:9200/documents`.
