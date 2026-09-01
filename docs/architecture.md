# Architecture

## System overview

```mermaid
flowchart LR
    subgraph Client
        UI[Streamlit UI]
    end

    subgraph Backend["FastAPI backend (backend/app)"]
        API["/chat, /rag/query"]
        Graph[LangGraph agent graph]
        Pipeline[RAG pipeline]
        API --> Graph
        API --> Pipeline
        Graph --> Pipeline
    end

    subgraph Data
        PG[(PostgreSQL + pgvector\ndocuments / parent_chunks / child_chunks / chat_messages)]
    end

    subgraph Mocks["Mock enterprise APIs"]
        WD[Mock Workday API]
        SN[Mock ServiceNow API]
    end

    Gemini[(Gemini API\nLLM + embeddings)]

    UI -->|HTTP| API
    Pipeline --> PG
    Pipeline --> Gemini
    Graph --> WD
    Graph --> SN
    Graph --> Gemini
```

## RAG pipeline (`backend/app/rag/pipeline.py`)

Every stage below is a real, separately-toggleable step -- see `.env.example`
for the `ENABLE_*` flags, and `backend/app/rag/pipeline.py` for the
orchestrating function. This is also exactly what the debug trace
(`trace: [...]` in API responses, or the Streamlit expander) shows you,
stage by stage, for any query.

```mermaid
flowchart TD
    Q[User query] --> MR[Metadata router\nrule-based category/country filters]
    MR --> QR{Query rewrite\nenabled?}
    QR -->|yes| QRW[LLM rewrites query]
    QR -->|no| EMB
    QRW --> EMB[Embed query\nGemini gemini-embedding-001]
    EMB --> HY{HyDE enabled?}
    HY -->|yes| HYP[LLM writes hypothetical answer,\nembed that instead]
    HY -->|no| QE
    HYP --> QE{Query expansion\nenabled?}
    QE -->|yes| QEV[LLM generates N phrasings,\nsearch with each, merge]
    QE -->|no| VS
    QEV --> VS[Vector search\npgvector + HNSW, cosine similarity]
    VS --> HS{Hybrid search\nenabled?}
    HS -->|yes| KS[Keyword search\nPostgreSQL FTS]
    KS --> RRF[Reciprocal Rank Fusion]
    HS -->|no| RR
    RRF --> RR{Reranking enabled?}
    RR -->|yes| CE[Cross-encoder reranks\ntop candidates]
    RR -->|no| PC
    CE --> PC{Parent/child\nenabled?}
    PC -->|yes| EXP[Expand child chunks\nto parent chunks, dedupe]
    PC -->|no| CTX
    EXP --> CTX[Assemble context\ntoken-budgeted]
    CTX --> ANS[Gemini generates answer\nwith citations]
```

## Agent graph (`backend/app/agents/graph.py`)

```mermaid
flowchart TD
    START([START]) --> SUP[supervisor\nembedding intent hint + LLM routing +\nmetadata filters]

    SUP -->|rag_hr / rag_it / rag_travel| RAG[rag_agent\nruns the RAG pipeline]
    SUP -->|workday / combined| EMP[employee_agent\nLLM tool-calling over\nget_employee / get_leave_balance / get_benefits]
    SUP -->|servicenow_troubleshoot,\nfirst turn| TS[it_troubleshoot\nRAG over IT docs,\nasks for confirmation]
    SUP -->|servicenow_troubleshoot,\nfollow-up turn| CR[it_check_resolution\nLLM judges RESOLVED/UNRESOLVED]

    EMP -->|intent == combined| RAG
    EMP -->|intent == workday| FINAL[final_answer]
    RAG -->|intent == combined| COMB[combine\nLLM merges Workday data + policy]
    RAG -->|intent != combined| FINAL

    COMB --> FINAL
    TS --> FINAL
    CR -->|resolved| FINAL
    CR -->|unresolved: LLM tool-calling\ncalls create_ticket| FINAL

    FINAL --> END([END])
```

Notes:
- `rag_agent` and `employee_agent` are each reached from two different
  routes; the conditional edges *after* those nodes check `state["intent"]`
  (set once by the supervisor and never overwritten) to decide whether to
  continue toward `combine` or go straight to `final_answer`.
- The `it_troubleshoot` / `it_check_resolution` split is what makes example
  query 5 ("try troubleshooting and create a ticket if it still doesn't
  work") a genuinely **conditional, multi-turn** workflow rather than always
  creating a ticket: `state["troubleshooting_resolved"]` persists across
  turns via the LangGraph `MemorySaver` checkpointer (keyed by
  `session_id`), and decides which of the two nodes handles the *next*
  message in the same session. See `backend/app/agents/it_agent.py`'s
  module docstring for the full walkthrough.
- Every node appends to `state["trace"]` (concatenated automatically via
  LangGraph's `operator.add` reducer -- see `backend/app/agents/state.py`),
  which is what the "User → Supervisor → Selected agent → Tool call → Tool
  parameters → Tool result → Next node → Final answer" debug view is built
  from.
