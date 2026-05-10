# RAG Assistant

A production-grade Retrieval-Augmented Generation system for querying arXiv research papers. The system automatically ingests daily papers, indexes them for search, and exposes an agentic question-answering API with observability built in.

---

## Features

- **Automated ingestion** — Apache Airflow DAG fetches, parses, and indexes new arXiv papers every weekday morning
- **Dual search modes** — BM25 keyword search and hybrid search (BM25 + KNN vector search fused with Reciprocal Rank Fusion)
- **Agentic RAG** — LangGraph-powered pipeline with guardrails, document grading, and automatic query rewriting on retrieval failure
- **Flexible LLM backend** — drop-in support for local Ollama models or OpenAI (controlled via environment variable, no code changes)
- **Telegram bot** — conversational interface to the RAG API
- **Gradio UI** — browser-based interface for interactive querying
- **Observability** — end-to-end request tracing via self-hosted Langfuse
- **Redis caching** — response caching to reduce latency on repeated queries

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Interfaces                             │
│   REST API (FastAPI)  │  Gradio UI  │  Telegram Bot             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       Agentic RAG (LangGraph)                   │
│   Guardrail → Retrieve → Grade → [Rewrite →] Generate          │
└──────┬──────────────────────────────────────────────────────────┘
       │                                        │
┌──────▼──────────┐                   ┌─────────▼──────────┐
│   OpenSearch    │                   │   LLM Backend      │
│  arxiv-papers   │                   │  Ollama / OpenAI   │
│  arxiv-papers-  │                   └────────────────────┘
│    chunks       │
└──────┬──────────┘
       │ synced by
┌──────▼──────────────────────────────────────────────────────────┐
│                      Airflow Ingestion DAG                      │
│  setup → fetch arXiv → store PostgreSQL → sync OpenSearch       │
│       → daily report → cleanup                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Two OpenSearch indices:**
- `arxiv-papers` — one document per paper, full text, BM25 keyword search
- `arxiv-papers-chunks` — ~10–20 chunks per paper with Jina embeddings, hybrid (BM25 + KNN) search

PostgreSQL is the source of truth. OpenSearch is the search layer. If the OpenSearch index is lost it can be rebuilt from PostgreSQL.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Agentic pipeline | LangGraph + LangChain |
| Search | OpenSearch 2.19 |
| Embeddings | Jina Embeddings v3 (1024-dim) |
| LLM (local) | Ollama |
| LLM (cloud) | OpenAI (gpt-4o-mini by default) |
| Ingestion scheduler | Apache Airflow |
| PDF parsing | Docling |
| Database | PostgreSQL 16 |
| Cache | Redis 7 |
| Observability | Langfuse (self-hosted) |
| UI | Gradio |
| Bot | python-telegram-bot |
| Containerisation | Docker Compose |

---

## Project Structure

```
src/
├── routers/                   # FastAPI route handlers
│   ├── ask.py                 # Simple RAG: retrieve + generate
│   ├── agentic_ask.py         # Agentic RAG: guardrail + grade + rewrite
│   ├── search.py              # BM25 keyword search
│   ├── hybrid_search.py       # Hybrid BM25 + KNN search
│   └── documents.py           # Document management
│
├── services/
│   ├── agents/                # LangGraph agentic pipeline
│   │   └── nodes/             # Individual graph nodes
│   ├── ingestion/             # arXiv fetch, PDF download, Docling parsing
│   ├── search_loaders/        # Write orchestrators: data → OpenSearch
│   │   ├── paper_loader.py    # PostgreSQL → OpenSearch (full papers)
│   │   ├── chunk_loader.py    # text → chunks → embeddings → OpenSearch
│   │   └── text_chunker.py    # Section-aware sliding-window chunking
│   ├── opensearch/            # Pure OpenSearch I/O layer
│   │   ├── client.py          # All OpenSearch operations
│   │   ├── query_builder.py   # Query DSL construction
│   │   └── index_config*.py   # Index schemas
│   ├── embeddings/            # Jina API client
│   ├── ollama/                # Ollama client + prompt builder
│   ├── openai/                # OpenAI client (same interface as Ollama)
│   ├── langfuse/              # Tracing client
│   └── cache/                 # Redis client
│
├── models/                    # SQLAlchemy models
├── schemas/                   # Pydantic request/response schemas
└── config.py                  # Settings (pydantic-settings, env-driven)

airflow/
└── dags/
    ├── ingestion_dag.py       # DAG definition (5-task pipeline)
    └── ingestion/
        └── tasks.py           # Task implementations
```

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- A [Jina API key](https://jina.ai/) for embeddings
- (Optional) An OpenAI API key if you prefer not to run Ollama locally

### 1. Clone and configure

```bash
git clone <repo-url>
cd rag-assistant
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required
JINA_API_KEY=your_jina_key

# LLM — pick one:
# Option A: local Ollama (default, no key needed)
OPENAI__ENABLED=false

# Option B: OpenAI
OPENAI__ENABLED=true
OPENAI__API_KEY=your_openai_key
OPENAI__MODEL=gpt-4o-mini

# Optional: Telegram bot
TELEGRAM__ENABLED=true
TELEGRAM__BOT_TOKEN=your_bot_token

# Optional: Langfuse tracing
LANGFUSE__ENABLED=true
LANGFUSE__PUBLIC_KEY=your_public_key
LANGFUSE__SECRET_KEY=your_secret_key
```

### 2. Start the stack

```bash
docker compose up -d
```

This starts: API, PostgreSQL, OpenSearch, Airflow, Ollama, Redis, Langfuse, and the OpenSearch Dashboard.

### 3. Pull an LLM model (if using Ollama)

```bash
docker exec rag-ollama ollama pull llama3.2:1b
```

### 4. Trigger the ingestion pipeline

Open the Airflow UI at [http://localhost:8080](http://localhost:8080) and manually trigger the `document_ingestion` DAG, or wait for it to run automatically on the next weekday at 06:00 UTC.

---

## API Reference

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Service health check |
| `/api/v1/ask` | POST | Simple RAG (retrieve + generate) |
| `/api/v1/ask-agentic` | POST | Agentic RAG with guardrails and query rewriting |
| `/api/v1/search/` | POST | BM25 keyword search |
| `/api/v1/hybrid-search/` | POST | Hybrid BM25 + KNN search with RRF |
| `/api/v1/documents/` | GET | List all ingested documents |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Example: agentic query

```bash
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key ideas behind attention mechanisms in transformers?"}'
```

### Example: hybrid search

```bash
curl -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "diffusion models for image generation", "size": 5}'
```

---

## Agentic Pipeline

The `/ask-agentic` endpoint runs a LangGraph graph with the following nodes:

```
START
  └─► guardrail         — rejects off-topic queries (non-research questions)
        ├─► out_of_scope — returns a polite refusal
        └─► retrieve     — hybrid search over the chunk index
              └─► grade_documents   — scores each chunk for relevance
                    ├─► rewrite_query — if no relevant chunks found, reformulates and retries retrieve
                    └─► generate_answer — synthesises an answer from graded chunks
                          └─► END
```

The rewrite loop prevents the system from generating hallucinated answers when retrieval fails — it reformulates the query up to a configurable number of times before giving up.

---

## Observability

Langfuse traces every agentic request: input query, retrieved chunks, grading decisions, LLM calls, and final answer. Access the dashboard at [http://localhost:3000](http://localhost:3000).

---

## Interfaces

### Gradio UI

```bash
python gradio_launcher.py
```

Opens a browser UI at [http://localhost:7860](http://localhost:7860).

### Telegram Bot

Set `TELEGRAM__ENABLED=true` and `TELEGRAM__BOT_TOKEN` in `.env`, then:

```bash
python -m src.telegram_bot
```

The bot connects to the `/ask-agentic` endpoint. Anyone with the bot link can use it.

---

## Ingestion Pipeline

The Airflow DAG (`document_ingestion`) runs five tasks in sequence:

1. **setup_environment** — verifies PostgreSQL and OpenSearch are reachable before spending time on fetching
2. **fetch_daily_papers** — calls the arXiv API, downloads PDFs, parses with Docling, stores in PostgreSQL
3. **sync_papers_to_opensearch** — reads parsed documents from PostgreSQL, loads them into OpenSearch
4. **generate_daily_report** — logs a structured summary (papers fetched, indexed, failures)
5. **cleanup_temp_files** — removes PDF files older than 30 days from `/tmp`

Schedule: Monday–Friday at 06:00 UTC. Each task retries twice with a 30-minute delay.
