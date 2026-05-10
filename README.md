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
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Interfaces                                 │
│                                                                              │
│   Gradio UI (port 7860)   REST API (port 8000)   Telegram Bot (long-poll)   │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     FastAPI  (src/main.py)       │
                    │  /ask  /ask-agentic  /search     │
                    │  /hybrid-search  /documents      │
                    └──┬──────────────────────────┬────┘
                       │                          │
         ┌─────────────▼──────────────┐  ┌───────▼────────────────────────┐
         │   Simple RAG               │  │   Agentic RAG (LangGraph)      │
         │                            │  │                                 │
         │   embed query (Jina)       │  │   START                        │
         │       ↓                    │  │     └► guardrail node          │
         │   hybrid search            │  │           ├► out_of_scope →END  │
         │   (OpenSearch RRF)         │  │           └► retrieve node     │
         │       ↓                    │  │                └► tool_retrieve │
         │   prompt builder           │  │                     └► grade   │
         │       ↓                    │  │                          ├► rewrite ─┐│
         │   LLM generate             │  │                          └► generate │ │
         │       ↓                    │  │                               └► END  │ │
         │   SSE stream               │  │                          (retry loop)─┘│
         └─────────┬──────────────────┘  └───────────────┬─────────────────────┘
                   │                                      │
                   └──────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────▼──────────────────────────┐
          │                   Service Layer                        │
          │                                                        │
          │  ┌─────────────────┐    ┌──────────────────────────┐  │
          │  │  OpenSearch      │    │   LLM Backend (duck-typed)│  │
          │  │                 │    │                           │  │
          │  │  arxiv-papers   │    │  OllamaClient             │  │
          │  │  (BM25, 1 doc   │    │    OR                     │  │
          │  │   per paper)    │    │  OpenAIClient             │  │
          │  │                 │    │  (identical interface,    │  │
          │  │  arxiv-papers-  │    │   swap via .env flag)     │  │
          │  │  chunks         │    └──────────────────────────┘  │
          │  │  (BM25+KNN,     │                                   │
          │  │  ~15 chunks     │    ┌──────────────────────────┐  │
          │  │  per paper,     │    │  Jina Embeddings API      │  │
          │  │  1024-dim vecs) │    │  (retrieval.passage vs    │  │
          │  └────────┬────────┘    │   retrieval.query tasks)  │  │
          │           │             └──────────────────────────┘  │
          │           │ synced by                                  │
          │  ┌────────▼──────────────────────────┐                │
          │  │  search_loaders/                   │                │
          │  │  paper_loader  → full-paper upsert │                │
          │  │  chunk_loader  → chunk + embed     │                │
          │  │                  + upsert pipeline │                │
          │  └────────┬──────────────────────────┘                │
          │           │ reads from                                  │
          │  ┌────────▼────────┐   ┌───────────┐  ┌────────────┐ │
          │  │  PostgreSQL     │   │   Redis    │  │  Langfuse  │ │
          │  │  (source of     │   │  (response │  │  (request  │ │
          │  │   truth for     │   │   cache)   │  │   tracing) │ │
          │  │   all papers)   │   └───────────┘  └────────────┘ │
          │  └────────▲────────┘                                  │
          └───────────┼───────────────────────────────────────────┘
                      │ written by
┌─────────────────────┴────────────────────────────────────────────────────┐
│                     Airflow Ingestion DAG (Mon–Fri 06:00 UTC)            │
│                                                                           │
│  Task 1: setup_environment   — verify DB + OpenSearch reachable          │
│      ↓                         (fail fast before wasting compute)        │
│  Task 2: fetch_daily_papers  — arXiv API → download PDFs → Docling parse │
│      ↓                         → upsert into PostgreSQL                  │
│  Task 3: sync_to_opensearch  — read PostgreSQL → bulk upsert OpenSearch  │
│      ↓                         (re-syncs all docs; upsert = safe to retry)│
│  Task 4: generate_daily_report — pull XCom stats from tasks 2 & 3       │
│      ↓                          → structured log (Slack/Datadog-ready)   │
│  Task 5: cleanup_temp_files  — delete PDFs older than 30 days from /tmp  │
│                                 (BashOperator: find -mtime +30 -delete)  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data flow

**Ingestion path** (background, scheduled):
arXiv API → PDFs → Docling parser → PostgreSQL → `paper_loader` (full-paper upsert) + `chunk_loader` (chunk → 1024-dim Jina embedding → OpenSearch)

**Query path** (real-time, per request):
query → Jina `retrieval.query` embedding → OpenSearch hybrid search (BM25 + KNN merged by RRF) → top-K chunks → LLM prompt → streaming answer

**Why two OpenSearch indices?**

| Index | Documents | Search type | Used by |
|---|---|---|---|
| `arxiv-papers` | 1 per paper, full text | BM25 keyword | `/search` endpoint |
| `arxiv-papers-chunks` | 10–20 per paper, 600-word chunks with 1024-dim embeddings | BM25 + KNN via RRF | `/hybrid-search`, `/ask`, `/ask-agentic` |

A full 10,000-word paper embedded as one vector produces a blurry average of all its topics. 600-word chunks let the search return the exact passage that answers the question. BM25 and KNN use separate term/vector indices — they cannot share statistics across these two fundamentally different document sizes.

**PostgreSQL is the source of truth.** OpenSearch is a derived search index. If lost, it can be fully rebuilt from PostgreSQL without re-fetching from arXiv.

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
├── main.py                         ← FastAPI app; lifespan manages all client lifecycles
├── config.py                       ← Pydantic Settings; all config from .env, fully typed
├── telegram_bot.py                 ← Long-polling Telegram bot → /ask-agentic
│
├── routers/                        ← HTTP layer only; no business logic
│   ├── ask.py                      ← POST /ask (JSON) and /stream (SSE token-by-token)
│   ├── agentic_ask.py              ← POST /ask-agentic → LangGraph graph
│   ├── search.py                   ← POST /search → BM25 on full papers
│   ├── hybrid_search.py            ← POST /hybrid-search → BM25+KNN+RRF on chunks
│   └── documents.py                ← CRUD /documents
│
├── services/
│   │
│   ├── agents/                     ← LangGraph agentic pipeline
│   │   ├── agentic_rag.py          ← AgenticRAGService: builds + compiles graph at startup
│   │   ├── state.py                ← AgentState dict (add_messages: append, not overwrite)
│   │   ├── context.py              ← Context dataclass: per-request dependency injection
│   │   ├── config.py               ← GraphConfig: top_k, model, thresholds, max retries
│   │   ├── models.py               ← GuardrailScoring, GradeDocuments, GradingResult, etc.
│   │   ├── prompts.py              ← GUARDRAIL_PROMPT, GRADE_DOCUMENTS_PROMPT, REWRITE_PROMPT
│   │   ├── tools.py                ← create_retriever_tool: LangChain tool → OpenSearch search
│   │   └── nodes/
│   │       ├── guardrail_node.py   ← LLM scores query 0-100; routes continue/out_of_scope
│   │       ├── retrieve_node.py    ← Emits tool_calls; LangGraph ToolNode executes them
│   │       ├── grade_documents_node.py  ← LLM: are retrieved chunks relevant? yes/no
│   │       ├── rewrite_query_node.py    ← LLM reformulates query (temp=0.3); retry loop
│   │       ├── generate_answer_node.py  ← LLM synthesises final answer from context
│   │       └── out_of_scope_node.py     ← Returns refusal for off-topic queries
│   │
│   ├── ingestion/                  ← arXiv → PDF → text → PostgreSQL
│   │   ├── fetcher.py              ← ArxivFetcher: Atom XML → DocumentMetadata list
│   │   ├── downloader.py           ← PDFDownloader: cache-aware, exponential backoff retry
│   │   ├── parser.py               ← PDFParser: Docling ML layout analysis → {full_text, sections}
│   │   └── orchestrator.py         ← IngestionOrchestrator: fetch→download→parse→upsert
│   │
│   ├── search_loaders/             ← Write orchestrators: move data INTO OpenSearch
│   │   ├── paper_loader.py         ← PostgreSQL → OpenSearch (full papers, BM25 index)
│   │   │                              sync_all_documents() called by Airflow task 3
│   │   ├── chunk_loader.py         ← ChunkLoader: text → chunks → Jina embed → OpenSearch
│   │   │                              process_paper() / process_papers_batch()
│   │   └── text_chunker.py         ← TextChunker: section-aware chunking with sliding-window
│   │                                  fallback; 600-word chunks, 100-word overlap
│   │
│   ├── opensearch/                 ← Pure OpenSearch I/O (no PostgreSQL, no Jina)
│   │   ├── client.py               ← OpenSearchClient: create_papers_index, upsert_paper,
│   │   │                              bulk_upsert_papers, bulk_upsert_chunks,
│   │   │                              search_papers, search_unified, health_check
│   │   ├── query_builder.py        ← PaperQueryBuilder (BM25 on papers), QueryBuilder (chunks)
│   │   │                              multi_match with field boosting (title^3, abstract^2)
│   │   ├── index_config.py         ← arxiv-papers mapping: text/date/keyword field types
│   │   └── index_config_hybrid.py  ← arxiv-papers-chunks mapping: knn_vector + RRF pipeline config
│   │
│   ├── embeddings/
│   │   └── jina_client.py          ← JinaEmbeddingsClient: persistent httpx.AsyncClient
│   │                                  embed_passages (retrieval.passage task)
│   │                                  embed_query    (retrieval.query task — different vector space)
│   │
│   ├── ollama/
│   │   ├── client.py               ← OllamaClient: generate_rag_answer (structured JSON output)
│   │   │                              generate_rag_answer_stream (plain text, SSE-compatible)
│   │   │                              get_langchain_model → ChatOllama
│   │   └── prompts.py              ← RAGPromptBuilder (system prompt from .txt file)
│   │                                  ResponseParser (3-level fallback: JSON → regex → plain text)
│   │
│   ├── openai/
│   │   └── client.py               ← OpenAIClient: identical interface to OllamaClient
│   │                                  get_langchain_model → ChatOpenAI
│   │                                  swap providers via OPENAI__ENABLED env var; no code changes
│   │
│   ├── langfuse/
│   │   ├── client.py               ← LangfuseTracer: start_trace, create_span, end_span, flush
│   │   │                              v2 REST API (compatible with self-hosted Langfuse v2.x)
│   │   └── tracer.py               ← RAGTracer: context manager for simple RAG tracing
│   │
│   └── cache/
│       └── client.py               ← Redis client for response caching
│
├── models/
│   └── document.py                 ← SQLAlchemy Document ORM → 'documents' table
│                                      fields: arxiv_id, title, authors, abstract,
│                                              full_text, sections, pdf_parsed, published_date
│
└── schemas/                        ← Pydantic contracts (API validation + serialisation)
    ├── api/ask.py                   ← AskRequest, AskResponse, AgenticAskResponse
    ├── api/search.py                ← SearchRequest, HybridSearchRequest, SearchHit
    ├── api/health.py                ← HealthResponse, ServiceStatus
    ├── indexing/chunks.py           ← TextChunk, ChunkMetadata (output of TextChunker)
    └── embeddings/jina.py           ← JinaEmbeddingRequest/Response

airflow/
└── dags/
    ├── ingestion_dag.py             ← DAG definition: 5 tasks, Mon–Fri 06:00 UTC,
    │                                   max_active_runs=1, retry×2 with 30min delay
    └── ingestion/
        └── tasks.py                 ← Task callables (lazy imports: Docling/Orchestrator
                                        loaded only at execution time, not DAG parse time)
                                        XCom: task 2 → task 3, tasks 2+3 → task 4
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
