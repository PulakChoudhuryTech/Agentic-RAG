# Enterprise Employee AI Assistant

A local, from-scratch learning project for understanding every layer of an
end-to-end **RAG + agentic AI system** -- not a production system, and not a
black box. Every retrieval stage and every agent decision is visible via a
DEBUG trace, and the code is written to be read, not just run.

It answers HR/IT/travel questions from company documents (RAG), looks up
live employee data from a mock Workday API, creates IT tickets through a
mock ServiceNow API, and combines the two when a question needs both --
routed by a LangGraph multi-agent graph with explicit state.

See [`docs/architecture.md`](docs/architecture.md) for diagrams and
[`docs/example_queries.md`](docs/example_queries.md) for what to try first.

## Tech stack

Python · FastAPI · LangChain · LangGraph · PostgreSQL + pgvector · Gemini
(LLM + embeddings) · PostgreSQL Full Text Search (keyword/BM25-style search)
· BGE-family cross-encoder reranker (`sentence-transformers`) · Streamlit ·
Docker Compose · pytest · optional LangSmith tracing.

## Project layout

```
backend/app/
  rag/          chunking, embeddings, vector search, keyword search, RRF,
                reranking, parent/child expansion, query rewrite/expansion,
                HyDE, intent classification, metadata routing, and the
                pipeline.py orchestrator that ties them all together
  agents/       LangGraph state, graph, supervisor, and the RAG/Employee/IT
                agents, plus the Workday/ServiceNow tools they call
  llm/          the one Gemini client + all prompt templates
  clients/      httpx clients for the mock Workday/ServiceNow APIs
  api/          FastAPI routes (/rag/query, /chat) and Pydantic schemas
  ingestion/    loads data/raw_docs/**, chunks, embeds, writes to Postgres
  trace/        the debug-trace mechanism used by both the pipeline and the graph
mocks/          standalone FastAPI apps: mock Workday and mock ServiceNow
db/             SQL migrations + a ~40-line custom migration runner
data/           sample HR/IT/travel documents, seed employee data, golden eval set
                (data/raw_docs/personal/ is for your own real PDFs -- gitignored)
eval/           retrieval metrics, RAGAS metrics, agent metrics, the eval CLI
ui/             Streamlit chat UI + RAG pipeline inspector
docs/           architecture diagrams, example queries, one doc per RAG concept
```

## Prerequisites

- Python 3.11+
- Docker (for Postgres/pgvector and the mock APIs)
- A [Gemini API key](https://ai.google.dev/) (used for both the LLM and the
  default embedding model)
- **Optional, only if you're ingesting scanned/image-based PDFs** (see
  "Personal documents" below): `tesseract` and `poppler` system binaries for
  the OCR fallback. On macOS: `brew install tesseract poppler`.

## Setup

```bash
# 1. Clone and create a virtualenv
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# The "eval" extra (RAGAS) is optional and separate -- pip install -e ".[eval]"
# -- since it pulls in a heavier dependency chain that can fail to build from
# source on some very new Python versions (unrelated to this project's code).

# 2. Configure environment
cp .env.example .env
# edit .env and set GEMINI_API_KEY

# 3. Start Postgres + the mock Workday/ServiceNow APIs
make db-up

# 4. Apply the database schema
make migrate

# 5. Ingest the sample documents (chunks + embeds data/raw_docs/**)
make ingest
```

## Running the app

**Recommended dev workflow**: run the backend and UI locally (not in
Docker) so you can set breakpoints, while Postgres and the mocks run in
Docker (from `make db-up` above):

```bash
make run-backend   # FastAPI on :8000
make run-ui        # Streamlit on :8501, in another terminal
```

Open http://localhost:8501. The "Chat" tab is the full agentic assistant;
the "RAG Inspector" tab calls `/rag/query` directly so you can study
retrieval in isolation.

**Alternative: everything in Docker** (no breakpoints, but one command):

```bash
docker compose --profile full up --build
```

## End-to-end testing lifecycle

This walks through the entire system once, in order, so you can see each
layer actually working before trusting the next one. Every step below is
independently verifiable -- don't skip the Postgres checks even if the API
calls "look fine"; the trace/citations only tell you what the app *claims*
happened, the SQL tells you what's *actually* in the database.

### 0. Confirm the infrastructure is up

```bash
docker compose ps
# expect: agentic_rag_postgres (healthy), agentic_rag_workday_mock, agentic_rag_servicenow_mock -- all "Up"

curl -s http://localhost:8001/health   # {"status":"ok","employees_loaded":5}
curl -s http://localhost:8002/health   # {"status":"ok"}
```

If these aren't up: `make db-up`.

### 1. Ingest documents and watch what happens

```bash
make ingest
```

Expected output: one line per source file under `data/raw_docs/**` (`.md`
files under `hr/`, `it/`, `travel/`, plus any `.pdf` files you've dropped
into `data/raw_docs/personal/` -- see below), then a summary line like
`done: 9 documents, 16 parent chunks, 77 child chunks (embedding
dims=768)`. If a PDF's text layer is nearly empty (a scanned/image-based
PDF), you'll see a `text layer nearly empty ... trying OCR` line before it
succeeds via `tesseract` -- or a `SKIPPING ...` line telling you to install
`tesseract`/`poppler` if OCR isn't available.

**Ingesting your own personal documents:** drop PDFs into
`data/raw_docs/personal/` (bills, insurance policies, e-tickets, etc. --
see that folder's `README.md`) and re-run `make ingest`. They're loaded
into a separate `personal` document category (`ROUTE_TO_CATEGORY["rag_personal"]`
in `backend/app/agents/rag_agent.py`), routed to by the supervisor exactly
like `rag_hr`/`rag_it`/`rag_travel`. `make ingest` is fully idempotent (it
`TRUNCATE`s and reloads all three tables every run), so re-running it after
adding more files is always safe.

### 2. Verify what actually landed in Postgres

```bash
docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "
  SELECT category, count(*) AS documents FROM documents GROUP BY category ORDER BY category;
"
docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "
  SELECT count(*) AS child_chunks, count(embedding) AS with_embedding FROM child_chunks;
"
# with_embedding should equal child_chunks -- every chunk got a real vector

docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "
  SELECT title, source_path, category, length(raw_text) AS chars FROM documents ORDER BY category, title;
"

# Spot-check the embedding dimension and that pgvector's index exists:
docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "
  SELECT vector_dims(embedding) FROM child_chunks LIMIT 1;
"   -- expect 768
docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "\di child_chunks_hnsw_idx"

# Country tagging (parent/child chunks of the parental leave policy's India
# section should be tagged, via ingestion/loader.py's tag_country()):
docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "
  SELECT metadata->>'country' AS country, count(*) FROM child_chunks
  WHERE metadata->>'country' IS NOT NULL GROUP BY 1;
"
```

If `documents` is empty or `with_embedding` is 0, ingestion didn't
actually run against a valid `GEMINI_API_KEY` -- check the terminal output
from `make ingest` for an API error before going further.

### 3. Exercise the RAG pipeline directly (bypasses agent routing)

```bash
make run-backend   # in one terminal; leave it running

# in another terminal:
curl -s -X POST 'http://localhost:8000/rag/query?debug=true' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the parental leave policy in India?"}' | jq
```

Check the response's `trace` array: you should see, in order, `pipeline`,
`metadata_router` (filters `{"category":"hr","country":"IN"}`),
`embeddings`, `vector_search`, `keyword_search`, `hybrid_search`,
`reranker`, `parent_child`, `context_assembly`, `llm` -- each with real
data (retrieved chunk text, cosine similarity scores, RRF scores, rerank
scores). `citations` should point at `hr/parental_leave_policy.md`. This is
the single best way to confirm the *retrieval* half of the system end to
end without also involving agent routing.

Try the same request with `"category": "personal"` in the body (or a query
your metadata router will auto-detect as personal, e.g. mentioning
"insurance" or "bill") to exercise the documents you ingested in step 1.

### 4. Exercise the full agentic system (`/chat`)

```bash
# RAG-only route
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-1", "message": "What is the parental leave policy in India?"}' | jq '.intent, .answer, .citations'

# Workday-only route (note employee_id)
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-2", "message": "How many vacation days do I have?", "employee_id": "E1002"}' | jq '.intent, .answer'
# expect "workday" and an answer mentioning 15 (see data/seed/employees.json)

# Combined route: Workday data + policy document
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-3", "message": "Am I eligible for parental leave based on my profile?", "employee_id": "E1001"}' | jq '.intent, .answer'

# Multi-turn conditional IT workflow: troubleshoot, then create a ticket if still broken
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-4", "message": "My VPN isn'"'"'t working. Try troubleshooting and create a ServiceNow ticket if it still doesn'"'"'t work.", "employee_id": "E1005"}' | jq '.intent, .answer'
# then, SAME session_id, reply that it's still broken:
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-4", "message": "I tried that, it still won'"'"'t connect.", "employee_id": "E1005"}' | jq '.intent, .answer'
# second answer should include an "INC..." ticket ID
```

See [`docs/example_queries.md`](docs/example_queries.md) for the full
annotated list (including personal-document queries) and
[`docs/architecture.md`](docs/architecture.md) for the graph diagram these
routes correspond to.

### 5. Verify chat history persisted (not just the in-memory graph state)

```bash
curl -s http://localhost:8000/chat/demo-4/history | jq '.messages[] | {role, content: .content[:80]}'
```

You should see all 4 messages (2 user, 2 assistant) from step 4's ticket
flow, each assistant turn carrying its own `trace`. This is reading from
`chat_messages` in Postgres, not LangGraph's in-memory checkpointer --
confirming persistence survives independently of the running process.

```bash
docker exec agentic_rag_postgres psql -U ragadmin -d agentic_rag -c "
  SELECT session_id, role, left(content, 60) FROM chat_messages ORDER BY created_at;
"
```

### 6. Same thing, visually, in the UI

```bash
make run-ui   # Streamlit on :8501, backend from step 3 still running
```

Open http://localhost:8501. Use the "Acting as" selector (top-right of the
Chat tab) to pick an employee, send the same queries as step 4, and expand
the debug trace under each response -- it's the same JSON as the `trace`
field above, rendered as expandable sections per pipeline/agent stage. The
"RAG Inspector" tab is the UI for step 3.

### 7. Compare configurations

Stop the backend (Ctrl-C), flip a flag in `.env` (try `ENABLE_RERANKING=false`
or `ENABLE_HYBRID_SEARCH=false`), `make run-backend` again, and resend one
of the step 3/4 queries. Diff the `trace` against your first run -- see
[`docs/example_queries.md`](docs/example_queries.md#comparing-configurations).

### 8. Run the automated tests and eval suite

```bash
make test             # unit + integration + e2e (needs the stack from steps 0-1 running)
make eval-retrieval    # Precision@K/Recall@K/MRR/NDCG against data/golden/golden_dataset.jsonl
make eval-agents       # routing/tool-selection/task-completion checks
```

`make test` runs against your currently-ingested documents and currently-set
feature flags, so results (especially retrieval metrics) will shift if you
add your own personal documents or toggle flags -- that's expected, and is
itself a way to observe the system's behavior changing.

## Seeing exactly what's happening (DEBUG mode)

Every response includes (when `DEBUG_MODE=true` in `.env`, the default, or
`?debug=true` on `/rag/query`) a `trace` array: an ordered list of
`{"node", "event", "data", "ts"}` steps covering every pipeline stage --
metadata filtering, query rewrite, vector search results, keyword search
results, RRF fusion, reranking, parent/child expansion, the assembled
context, and the final answer -- or, for `/chat`, the agent's routing
decision, which agent ran, every tool call and its result, and the final
answer. The Streamlit UI renders this as an expandable trace under every
message. Past sessions stay inspectable too: `GET /chat/{session_id}/history`
replays the full conversation with each turn's trace, from Postgres.

## Feature flags: turn things on/off and compare

Every "advanced" retrieval feature is a plain boolean in `.env`
(`backend/app/config.py`), read once at startup:

| Flag | Default | What it does |
|---|---|---|
| `ENABLE_HYBRID_SEARCH` | `true` | Combine vector + keyword search via RRF, instead of vector-only |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranks the fused candidates |
| `ENABLE_PARENT_CHILD` | `true` | Expand matched child chunks to their parent chunk for more context |
| `ENABLE_QUERY_REWRITE` | `false` | LLM rewrites the query before retrieval |
| `ENABLE_QUERY_EXPANSION` | `false` | LLM generates multiple query phrasings, searched and merged |
| `ENABLE_HYDE` | `false` | Embed an LLM-generated hypothetical answer instead of the raw query |
| `ENABLE_METADATA_ROUTING` | `true` | Rule-based category/country filters applied to search |
| `ENABLE_EMBEDDING_INTENT_CLASSIFIER` | `true` | Cheap embedding-similarity routing hint shown alongside the LLM supervisor's decision |

Flip a flag, restart the backend, resend the same query, and diff the debug
trace. `GET /config` (shown in the Streamlit sidebar) shows what's currently
active.

## Key components, explained

Each RAG concept has its own short doc under `docs/concepts/`:

- [Reciprocal Rank Fusion (RRF)](docs/concepts/rrf.md)
- [HNSW](docs/concepts/hnsw.md)
- [PostgreSQL FTS vs. real BM25](docs/concepts/fts_vs_bm25.md) -- **read this
  one**: this project deliberately does not use Elasticsearch/OpenSearch,
  and the keyword-search layer is built to be swappable later
- [Cross-encoder reranking](docs/concepts/reranking.md)
- [Parent/child retrieval](docs/concepts/parent_child_retrieval.md)
- [HyDE](docs/concepts/hyde.md)

And [`docs/architecture.md`](docs/architecture.md) for the RAG pipeline and
agent graph diagrams.

## Design choices worth knowing about

- **Raw SQL for the retrieval core.** `rag/vector_store.py`,
  `rag/keyword_search.py`, and `rag/hybrid_search.py` are plain `psycopg` +
  hand-written SQL -- not LangChain's `PGVector`/`Retriever` classes, which
  would hide the exact SQL, cosine-similarity math, and RRF math this
  project exists to show. LangChain/LangGraph are used where the
  abstraction itself is pure plumbing or the thing being taught:
  `ChatGoogleGenerativeAI` as the LLM wrapper, `bind_tools` for agent tool
  calling, `StateGraph` for graph orchestration.
- **No Alembic.** `db/migrate.py` is a ~40-line custom runner for 6 SQL
  files -- simpler to read end-to-end than Alembic's revision DSL at this
  scale.
- **`rag/pipeline.py` is the file to read first.** It's one straight-line
  function with explicit `if settings.enable_x:` branches per feature --
  no strategy pattern, no plugin registry.

## Evaluation

```bash
make eval-retrieval   # Precision@K, Recall@K, MRR, NDCG@K (fast, free, no LLM judge)
make eval-agents      # routing / tool selection / tool execution / task completion (fast, free)
make eval-ragas       # RAGAS faithfulness + answer relevancy, small subset (slower, uses Gemini as judge)
make eval-all         # retrieval + agents (ragas is excluded from "all" -- see eval/run_eval.py)
```

Each run writes a timestamped JSON file to `eval/results/` and regenerates
[`docs/eval_report.md`](docs/eval_report.md). The golden dataset is
`data/golden/golden_dataset.jsonl` -- small and hand-written on purpose, so
you can read every case.

Retrieval metrics (Precision@K/Recall@K/MRR/NDCG) are implemented directly
in `eval/retrieval_metrics.py` -- simple enough to be worth seeing plainly.
Faithfulness/answer relevancy use [RAGAS](https://github.com/explodinggecko/ragas)
since those need an LLM judge, which RAGAS already does well.

## Tests

```bash
make test   # backend/tests (unit + integration + e2e) + eval threshold checks
```

Unit tests (`backend/tests/unit/`) never touch the network or a real
database. Integration and e2e tests need the full stack running (`make
db-up && make migrate && make ingest`, plus a valid `GEMINI_API_KEY`) and
will skip cleanly (not fail) if it isn't available -- see
`backend/tests/conftest.py`.

## Optional: LangSmith tracing

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` in `.env` to get
full LangChain/LangGraph traces in [smith.langchain.com](https://smith.langchain.com).
Off by default; nothing LangSmith-related is contacted unless you enable it.

## Troubleshooting

- **"role ... does not exist" / can't connect to Postgres on startup**: this
  almost always means something else on your machine is already listening on
  port `55432` (docker-compose's mapped Postgres port) and you're connecting
  to that instead of this project's container. Run `lsof -i :55432` to check,
  and change the port mapping in `docker-compose.yml` (and `DATABASE_URL` in
  `.env`) if it's taken.
- **Backend fails to start with a `Settings` validation error**: `.env` is
  missing a required value -- almost always `GEMINI_API_KEY`. Check
  `.env.example` for the full list of required vs. optional vars.
- **`make ingest` is slow / hits rate limits**: it's making one Gemini
  embedding call per document section (batched per parent chunk, not per
  child chunk) -- with the sample doc set this is well under free-tier
  limits, but if you add many more documents, batch further in
  `ingestion/ingest.py`.
- **`make ingest` prints `SKIPPING <file>.pdf: no usable text layer and OCR
  failed`**: that PDF has no extractable text (common for scanned documents
  or e-tickets/boarding passes) and the OCR fallback couldn't run --
  install `tesseract` and `poppler` (`brew install tesseract poppler` on
  macOS) and re-run. See `ingestion/loader.py`'s module docstring for how
  the two-tier extraction (text layer, then OCR) works.
- **`pip install -e ".[eval]"` fails building `scikit-network`**: this is a
  RAGAS transitive dependency failing to compile its C extension on some
  newer Python versions/toolchains, unrelated to this project's own code.
  The base install (`pip install -e ".[dev]"`) and `make eval-retrieval`/
  `make eval-agents` don't need it; only `make eval-ragas` does. If you hit
  this, either use a slightly older Python (3.11-3.12 tends to have
  prebuilt wheels available) or skip `make eval-ragas`.

## Known simplifications (this is a learning project, not production)

- No auth: `employee_id` is passed directly (via the Streamlit "acting as"
  selector, or the API's `employee_id` field) rather than derived from a
  real identity system.
- LangGraph's `MemorySaver` checkpointer is in-process memory only --
  multi-turn conversation state (including the IT troubleshoot/ticket flow)
  is lost on backend restart. `chat_messages` in Postgres persists the
  visible history/trace regardless.
- PostgreSQL FTS is not BM25 -- see
  [`docs/concepts/fts_vs_bm25.md`](docs/concepts/fts_vs_bm25.md).
- Token counts are estimated by word count, not a real tokenizer.
- The mock Workday API is read-only, in-memory, seeded from
  `data/seed/employees.json`. The mock ServiceNow API persists tickets to a
  local SQLite file.
- `data/raw_docs/personal/*.pdf` and the DB rows/embeddings derived from
  them are real personal data, not sample content -- they're gitignored,
  and deliberately excluded from `data/golden/golden_dataset.jsonl` (which
  *is* committed) so no personal facts end up tracked in version control.
  If you want retrieval metrics against your own personal docs, add cases
  to a local, gitignored copy of the golden dataset rather than the
  tracked one.
