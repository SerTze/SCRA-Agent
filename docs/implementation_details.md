# SCRA Implementation Details

> Snapshot as of **2026-02-12** — full context for any agent or developer continuing work on this codebase.

---

## 1. What This Project Is

**SCRA (Self-Correcting Regulatory Agent) v4.0** — a compliance-aware Q&A agent that answers questions about the **EU AI Act** (Regulation (EU) 2024/1689). It retrieves evidence from a vector store, generates cited answers via LLM, then self-audits via grounding and compliance grading loops. If the answer is deemed hallucinated or non-compliant, it rewrites the query and retries — up to 3 times — before falling back to web search.

---

## 2. Architecture Decisions

### 2.1 Clean Architecture (Hexagonal / Ports & Adapters)

The codebase is split into 4 strict layers with enforced dependency rules:

```
Domain  ←  Application  ←  Infrastructure
                ↑
          Presentation
```

| Layer | Path | Allowed imports | Purpose |
|---|---|---|---|
| **Domain** | `src/domain/` | Only Python stdlib + Pydantic V2 | Models, protocols (interfaces), exceptions |
| **Application** | `src/application/` | Domain only | Business logic services |
| **Infrastructure** | `src/infrastructure/` | Domain + external libs | Adapters implementing domain protocols |
| **Presentation** | `src/presentation/` | All layers | LangGraph workflow + FastAPI endpoints |

**Why:** The spec demanded Clean Architecture. The strict layer boundaries ensure domain logic is testable without any infrastructure, and adapters can be swapped without touching business logic.

### 2.2 Dependency Injection via `Container`

All wiring lives in `src/container.py`. A single `Container` instance is created at app startup and holds every adapter and service. Nodes in the LangGraph workflow receive services via closure injection — they never instantiate clients themselves.

**Why:** The spec explicitly prohibited global singletons and required that clients never be created inside nodes.

### 2.3 Protocol-Based Interfaces

Domain protocols (`LLMPort`, `RetrieverPort`, `RerankerPort`, `WebSearchPort`, `VectorStorePort`) are `typing.Protocol` classes with `@runtime_checkable`. Adapters satisfy these protocols structurally (duck typing) without explicit inheritance.

**Why:** Keeps the domain layer pure Python with no infrastructure imports. Makes mocking trivial in tests.

---

## 3. Domain Layer (`src/domain/`)

### 3.1 Models (`models.py`)

| Model | Purpose |
|---|---|
| `EvidenceChunk` | A single piece of evidence with validated `source_id`, `source_type` (Literal: `primary_legal` / `secondary_summary` / `web_fallback`), `content`, `metadata` (Dict[str,str]), `relevance_score` |
| `GroundingResult` | Structured result from the grounding grader: `score` (Literal: `grounded` / `partial` / `hallucinated`), `reasoning` |
| `GenerationResult` | Structured result from answer generation: `answer`, `cited_sources` (list of source IDs; empty list signals insufficient evidence, triggering web fallback) |
| `ComplianceAnalysis` | Result of compliance grading: `is_compliant`, `risk_flags`, `reasoning` |
| `GraphState` | The Pydantic state object flowing through LangGraph: `question`, `original_question`, `documents`, `generation`, `cited_sources`, `grounding_score` (4-way Literal), `compliance_analysis`, `loop_step`, `max_retries`, `fallback_active` |
| `RetrievalSettings` | Value object for retrieval hyper-parameters: `TOP_K_RETRIEVAL=25`, `TOP_K_FINAL=5`, `PRIMARY_SOURCE_BOOST=1.2` |

### 3.2 Citation Regex (Single Source of Truth)

Two compiled regex constants live in `models.py`:

- **`CITATION_PATTERN`** — matches inline citations like `[EUAI_Art5_Chunk2]`, `[WEB_example.com_a1b2c3d4]`, etc. Used by `CitationService` and the API to extract citations from LLM output.
- **`SOURCE_ID_PATTERN`** — anchored version for validating `EvidenceChunk.source_id` via `field_validator`.

Supported formats:
- `EUAI_Art{N}_Chunk{N}` / `EUAI_Art{N}_Sec{X}_Chunk{N}` (articles)
- `EUAI_Rec{N}_Chunk{N}` (recitals)
- `EUAI_Annex{X}_Chunk{N}` (annexes)
- `EUAI_Page{N}_Chunk{N}` (page fallback)
- `EUAI_File{slug}_Chunk{N}` (file fallback)
- `WEB_{domain}_{hex_hash}` (web search fallback)

### 3.3 Exceptions (`exceptions.py`)

Rich hierarchy rooted at `SCRAError`. Notable:
- `LLMResponseParsingError` — stores `raw_response` for diagnostics; triggers fail-fast on invalid JSON.
- `CitationValidationError` — carries `missing_inline` and `missing_sources` lists.
- `LatencyBudgetExceeded` — stores elapsed vs budget for observability.

---

## 4. Application Layer (`src/application/services/`)

Four single-responsibility services — **no logic is duplicated** across workflow nodes:

### 4.1 `RetrieverService`

5-step retrieval pipeline:
1. **Vector retrieval** via `RetrieverPort` (ChromaDB) — top 25
2. **Web fallback** via `WebSearchPort` (Tavily) — only when `use_web_fallback=True`
3. **Reranking** via `RerankerPort` (Cohere) — narrows to top 5
4. **Source-priority boost** — `primary_legal` chunks get 1.2× score multiplier; articles/annexes get an additional 5% sub-boost over recitals
5. **Sort** by boosted score descending

### 4.2 `CitationService`

Stateless service enforcing the bi-directional citation contract:

**Structured path (preferred):** `validate_structured()` takes a list of cited source IDs from `GenerationResult.cited_sources` and checks that all IDs exist in the document set. Empty lists are accepted (caller decides fallback).

**Legacy regex path:** `validate()` provides backward compatibility for plain-text LLMs:
- Every inline `[SOURCE_ID]` must appear in the `Sources:` block
- Every `Sources:` entry must be used inline
- All cited IDs must exist in the provided document set

**Key fix applied:** `extract_inline_citations()` only scans text *before* the `Sources:` block to avoid double-counting source block entries as inline citations.

### 4.3 `GradingService`

Two LLM-based grading steps:
- **Grounding:** returns `grounded` / `partial` / `hallucinated` / `unknown`
- **Compliance:** returns `ComplianceAnalysis` with `is_compliant`, `risk_flags`, `reasoning`

**Structured output (preferred path):** Uses `generate_structured()` with provider-enforced schemas (`GroundingResult`, `ComplianceAnalysis`) so the LLM is constrained to return valid Pydantic objects. Falls back to legacy `generate()` + JSON-parse path when the adapter does not support `generate_structured()` or when structured output raises an exception.

**Legacy fallback:** `_parse_json()` strips markdown code fences before parsing. All failures degrade gracefully (return `unknown` or non-compliant with diagnostic flags), never crash.

### 4.4 `GenerationService`

Builds the RAG prompt with a hardcoded system prompt (citation rules + answer format) and evidence block. Key features:

- **Input sanitization:** `_sanitize_input()` strips known prompt-injection patterns (`ignore previous instructions`, `you are now`, `system prompt:`, etc.) via regex before the user question reaches the LLM prompt.
- **Evidence truncation:** The evidence block is capped at `_MAX_EVIDENCE_CHARS` (12,000 chars ≈ 3,000 tokens) to prevent context-window overflow with large document sets.
- **Structured output (preferred):** Uses `generate_structured()` with `GenerationResult` schema when the adapter supports it; falls back to plain text generation wrapped in `GenerationResult(answer=raw, cited_sources=[])`.
- **Query rewriting:** `rewrite_question()` provides query expansion during retry loops.

---

## 5. Infrastructure Layer (`src/infrastructure/`)

### 5.1 Adapters

| Adapter | Implements | External Service | Notes |
|---|---|---|---|
| `GroqAdapter` | `LLMPort` | Groq (Llama) | Uses `langchain-groq` `ChatGroq` with `ainvoke()`. Caches `with_structured_output` runnables per schema in `_structured_cache` for performance. Tenacity retry (3 attempts, exponential backoff). |
| `OpenAIAdapter` | `LLMPort` | OpenAI | Uses `langchain-openai` `ChatOpenAI` with `ainvoke()`. Same structured output caching and retry strategy as `GroqAdapter`. |
| `CohereAdapter` | `RerankerPort` | Cohere Rerank v3.5 | Uses `cohere.AsyncClientV2`; maps results back via `model_copy(update=...)`. Tenacity retry. |
| `ChromaAdapter` | `RetrieverPort` + `VectorStorePort` | ChromaDB (local, persistent) | Cosine similarity; `PersistentClient`; skips chunks with invalid source_ids; upserts with composite IDs (source_id + content_hash). Uses `asyncio.to_thread` for all blocking ChromaDB calls. |
| `TavilyAdapter` | `WebSearchPort` | Tavily | `search_depth="advanced"`; generates `WEB_{domain}_{sha256[:8]}` source IDs. Tenacity retry. |

### 5.2 Ingestion Pipeline (`ingestion.py`)

Downloads and chunks the **full EU AI Act** from the EU Publications Office.

#### Download Strategy (Critical Fix)

The original EUR-Lex URL (`eur-lex.europa.eu/legal-content/EN/TXT/HTML/...`) is **blocked by AWS WAF** — returns HTTP 202 with an empty body or a JavaScript challenge page. This was discovered during verification testing.

**Solution:** The pipeline uses the **EU Publications Office CELLAR** endpoint as primary source:
```
https://publications.europa.eu/resource/cellar/dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0006.03/DOC_1
```
This returns the full XHTML directly (1.25 MB, 116 articles, 13 annexes) with no WAF blocking.

**Fallback chain for download:**
1. Local file (`eu_ai_act.html` or custom path) — for offline/CI use
2. CELLAR URL (primary remote)
3. EUR-Lex URL (secondary remote, may fail due to WAF)

#### Parsing Strategy

The XHTML is parsed with `lxml-xml` (auto-detected) or `lxml`/`html.parser` fallback:

1. **Articles** — regex finds `Article N` headings in full text; extracts text between consecutive article boundaries; deduplicates by article number
2. **Recitals** — `(N)` pattern in the preamble section; extracts text between consecutive recital matches
3. **Annexes** — `ANNEX {roman/arabic}` pattern; walks up DOM to find surrounding content

All text is split into ≤2000-char chunks on paragraph boundaries via `_split_text()`.

#### Ingestion Resilience (4-Level Fallback)

If structured parsing yields 0 chunks:
1. Article regex detection
2. Recital numbering detection
3. Page-based chunking (`EUAI_PageN_ChunkN`)
4. File-based chunking (`EUAI_File..._ChunkN`)

The pipeline **never crashes** — worst case produces a single placeholder chunk.

#### Verified Output (2026-02-11)

Chunk counts can vary slightly as the upstream HTML changes and the chunking heuristics evolve.

| Metric | Value |
|---|---|
| Total chunks | ~700 (last verified run ingested 704) |
| Invalid source_ids | 0 |

### 5.3 Telemetry (`telemetry.py`)

Sets up OpenTelemetry with `TracerProvider`. Uses `SimpleSpanProcessor` with `ConsoleSpanExporter` by default. When `OTEL_EXPORTER_ENDPOINT` is configured, switches to `BatchSpanProcessor` with `OTLPSpanExporter`. FastAPI is auto-instrumented via `FastAPIInstrumentor`.

---

## 6. Presentation Layer (`src/presentation/`)

### 6.1 LangGraph Workflow (`workflow.py`)

Self-correcting agent loop implemented as a compiled `CompiledStateGraph`:

```
retrieve → generate → grade (grounding + compliance concurrently via asyncio.gather) → [decide]
  ├─ grounded + compliant → END
  ├─ no cited_sources (evidence insufficient) → web_fallback (or error if already in fallback)
  ├─ partial / hallucinated / non-compliant (retries left) → rewrite → retrieve (loop)
  ├─ max_retries reached, no fallback yet → web_fallback → retrieve
  └─ max_retries + fallback already active → error → END
```

**Node pattern:** Each node is created by a factory function (`_make_*_node()`) that returns an async closure over injected services. Nodes are thin — they delegate to application services and return state patches (dicts).

**Grade node optimization:** Grounding and compliance grading run concurrently via `asyncio.gather()`, halving the latency of the grading phase. Citation validation runs inline after both complete; invalid citations downgrade a "grounded" score to "partial".

**Decision function:** `_decide_after_grounding(state)` routes based on: (1) `cited_sources` emptiness (triggers fallback directly), (2) `grounding_score`, (3) `compliance_analysis.is_compliant`, (4) `loop_step` vs `max_retries`, and (5) `fallback_active`.

### 6.2 FastAPI App (`api.py`)

App factory pattern (`create_app(settings=None) -> FastAPI`):

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Full self-correcting agent loop; returns answer, grounding score, compliance analysis, cited sources, latency, fallback status |
| `/health` | GET | ChromaDB connectivity check + document count (async-safe via `asyncio.to_thread`) |
| `/ingest` | POST | Triggers EU AI Act download + parsing + ChromaDB storage |

Request validation: question must be 3–2000 chars.

**Lifespan management:** Uses `asynccontextmanager`-based lifespan (not deprecated `@app.on_event`). Shutdown flushes the OpenTelemetry tracer provider with error protection (non-fatal if it fails).

### 6.3 Latency Budget Middleware (`middleware.py`)

Pure ASGI middleware (not `BaseHTTPMiddleware`) — works correctly with streaming responses and background tasks. Wraps every non-exempt request in `asyncio.wait_for(app(scope, receive, send_with_latency), timeout=budget)`. If the timeout fires, returns HTTP 504 with JSON detail. Exempt paths (`/ingest`, `/health`) bypass the budget. Always sets `X-Latency-Ms` response header. Normalises paths with trailing-slash tolerance.

---

## 7. Configuration (`src/config/settings.py`)

Single `Settings(BaseSettings)` class with `.env` file support. Key defaults:

| Setting | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Select LLM backend (`groq` or `openai`) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model (when `LLM_PROVIDER=openai`) |
| `GROQ_TEMPERATURE` | `0.0` | Deterministic output |
| `COHERE_RERANK_MODEL` | `rerank-v3.5` | Reranker model |
| `TOP_K_RETRIEVAL` | `25` | Initial retrieval count |
| `TOP_K_FINAL` | `5` | Post-reranking count |
| `PRIMARY_SOURCE_BOOST` | `1.2` | Score multiplier for primary legal sources |
| `LATENCY_BUDGET_SECONDS` | `10.0` | Max acceptable request latency |
| `MAX_RETRIES` | `3` | Self-correction loop limit |
| `EUR_LEX_URL` | CELLAR URL | Primary document source |

---

## 8. Testing (`tests/`)

### Test Files

| File | Tests | What it covers |
|---|---|---|
| `test_domain_models.py` | 32 | EvidenceChunk validation, GraphState defaults, GenerationResult, citation regex (parametrized), RetrievalSettings |
| `test_citation_service.py` | 12 | Structured validation, inline extraction, legacy validation, unknown document |
| `test_grading_service.py` | 15 | Grounding: structured, structured-fallback, empty-docs. Legacy grounding. Compliance: structured, legacy. JSON parsing. |
| `test_retriever_service.py` | 4 | Basic retrieval flow, web fallback, empty retrieval, source boost ordering |
| `test_workflow.py` | 42 | Decision routing (11), quality ranking (6), stall detection (5), evidence-insufficient routing (6), evidence-insufficient detection (14) |
| `test_eval_scoring.py` | 14 | Relevance scoring (5), citation validity (5), citation source (4) |
| `test_api.py` | 8 | Health endpoint, request validation, mocked workflow, cache integration |
| `test_chunking.py` | 19 | Fixed-size splitting, paragraph splitting, metadata prepend, edge cases |
| `test_cache.py` | 10 | LRU + TTL + semantic similarity, case normalisation, eviction |
| `test_rate_limit.py` | 7 | Token bucket, TTL eviction, interval guard |
| `test_integration.py` | 3 (skipped) | Live Groq/Cohere/Tavily calls, gated behind `RUN_LIVE_TESTS=1` |

**Total: 166 passed, 3 skipped** (as of 2026-02-20)

### Test Strategy

- **Unit tests** mock all external APIs via `unittest.mock.AsyncMock`
- **Integration tests** gated behind `RUN_LIVE_TESTS=1` env var
- **Fixtures** in `conftest.py` provide reusable `Settings`, sample chunks, and sample state
- **pytest-asyncio** with `asyncio_mode="auto"` for async test support

---

## 9. Bugs Found & Fixed

### 9.1 Citation Double-Counting

**Problem:** `CitationService.extract_inline_citations()` scanned the *entire* text including the `Sources:` block, so citations listed in Sources were also counted as inline citations. This made the "sources not used inline" validation always pass even when no inline citations existed.

**Fix:** Changed `extract_inline_citations()` to only scan text *before* the `Sources:` marker:
```python
sources_idx = text.find("Sources:")
body = text[:sources_idx] if sources_idx != -1 else text
```

### 9.2 EUR-Lex WAF Blocking

**Problem:** The original EUR-Lex HTML URL returns HTTP 202 with empty body — AWS WAF blocks automated requests.

**Fix:** Switched primary download source to EU Publications Office CELLAR endpoint which serves the full XHTML directly. Added multi-URL fallback chain and local file support. Added browser-like HTTP headers.

### 9.3 XML Parsed as HTML Warning

**Problem:** The CELLAR document is XHTML (XML), but was being parsed with `lxml` (HTML mode), triggering `XMLParsedAsHTMLWarning`.

**Fix:** Auto-detect XHTML by checking for `<?xml` or `xmlns=` in the first 500 chars, then use `lxml-xml` parser. Suppressed the warning via `warnings.filterwarnings`.

---

## 10. Environment Setup

- **Conda env:** `scra` with Python 3.11+
- **Activation:** `conda activate scra`
- **Dependencies:** installed via `pip install -r requirements.txt` in the conda env
- **Note:** `conda run -n <env>` can be used to run commands without activating the environment

---

## 11. What's Needed to Run

1. **API Keys** in `.env`: `GROQ_API_KEY`, `COHERE_API_KEY`, `TAVILY_API_KEY`
2. **Ingest** the EU AI Act: `POST /ingest` or `python verify_ingestion.py`
3. **Start server:** `uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000` (or `uvicorn src.main:app` for non-factory mode)

No manual file downloads needed — the ingestion pipeline fetches the EU AI Act automatically from the CELLAR endpoint.

---

## 12. File Inventory

```
src/
├── __init__.py
├── main.py                              # Entry point: create_app() → app (supports --factory mode)
├── container.py                         # DI composition root (cached_property for lazy init)
├── config/
│   ├── __init__.py
│   └── settings.py                      # Pydantic Settings from .env
├── domain/
│   ├── __init__.py
│   ├── models.py                        # EvidenceChunk, GraphState, ComplianceAnalysis, citation regex
│   ├── protocols.py                     # LLMPort, RetrieverPort, RerankerPort, WebSearchPort, VectorStorePort
│   └── exceptions.py                    # SCRAError hierarchy
├── application/
│   ├── __init__.py
│   ├── workflow.py                      # LangGraph self-correcting agent loop (CompiledStateGraph)
│   ├── prompts.py                       # Central prompt templates
│   ├── cache.py                         # LRU response cache + semantic similarity (ONNX MiniLM)
│   └── services/
│       ├── __init__.py
│       ├── retriever_service.py         # Retrieve + rerank + boost pipeline
│       ├── citation_service.py          # Bi-directional citation validation
│       ├── grading_service.py           # Grounding + compliance grading (structured output + legacy fallback)
│       ├── generation_service.py        # RAG prompt building + answer generation + injection sanitization
│       └── evidence_builder.py          # Shared evidence block builder for LLM prompts
├── infrastructure/
│   ├── __init__.py
│   ├── groq_adapter.py                  # Groq/Llama 3 70B (LLMPort)
│   ├── openai_adapter.py                # OpenAI (LLMPort)
│   ├── base_llm_adapter.py              # Shared base: structured output cache, token tracking, audit log
│   ├── cohere_adapter.py                # Cohere rerank v3.5 (RerankerPort)
│   ├── chroma_adapter.py                # ChromaDB retrieval + storage (asyncio.to_thread)
│   ├── tavily_adapter.py                # Tavily web search (WebSearchPort)
│   ├── ingestion.py                     # EUR-Lex XHTML → EvidenceChunk pipeline
│   └── telemetry.py                     # OpenTelemetry setup
└── presentation/
    ├── __init__.py
    ├── api.py                           # FastAPI (/query, /health, /ingest) + error-protected lifespan
    ├── middleware.py                    # Pure ASGI latency budget middleware
    └── rate_limit.py                    # Per-IP token bucket rate limiter

tests/
├── __init__.py
├── conftest.py                          # Shared fixtures + live_test gate
├── test_domain_models.py                # 32 tests
├── test_citation_service.py              # 12 tests
├── test_grading_service.py               # 15 tests
├── test_retriever_service.py             # 4 tests
├── test_workflow.py                      # 42 tests
├── test_eval_scoring.py                  # 14 tests
├── test_api.py                           # 8 tests
├── test_chunking.py                      # 19 tests
├── test_cache.py                         # 10 tests
├── test_rate_limit.py                    # 7 tests
└── test_integration.py                  # 3 tests (skipped without RUN_LIVE_TESTS=1)

docs/
├── compliance_matrix.md                 # 15 compliance controls mapped to code + tests
└── implementation_details.md            # This file

Root files:
├── .env.example                         # API key template
├── .gitignore
├── Makefile                             # env, test, serve, ingest targets
├── pyproject.toml                       # pytest + ruff config
├── README.md
├── requirements.txt
└── verify_ingestion.py                  # Standalone ingestion smoke test
```
