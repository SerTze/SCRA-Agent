---
project_name: "SCRA - Self-Correcting Regulatory Agent"
one_liner: "A compliance-aware EU AI Act QA system that uses retrieval, citation-grounded generation, and self-auditing loops before returning answers."
domain: ["legal-tech", "regtech", "compliance"]
project_type: ["RAG", "agentic-workflow", "evaluation-harness"]
my_role: "unknown"
status: "unknown"
timeframe: "Feb 2026 snapshot"
stack: ["Python 3.11", "FastAPI", "LangGraph", "LangChain", "ChromaDB", "Groq/OpenAI", "Cohere Rerank", "Tavily", "OpenTelemetry", "Docker"]
data_sources: ["EU AI Act HTML/XHTML from EU Publications Office CELLAR with EUR-Lex fallback", "Tavily web pages for fallback evidence", "golden_dataset.json evaluation set"]
deployment: ["Docker", "docker-compose", "local uvicorn"]
keywords: ["EU AI Act", "Regulation 2024/1689", "regulatory QA", "RAG", "LangGraph", "citation validation", "grounding", "compliance grading", "ChromaDB", "Cohere reranking", "web fallback", "FastAPI"]
---

# SCRA - Self-Correcting Regulatory Agent

## What it is
SCRA is a regulatory QA system focused on the EU AI Act (Regulation (EU) 2024/1689). It is built for users who need source-backed answers with explicit citations, not just fluent summaries. The system combines retrieval, generation, grading, and fallback routing in one loop.

## Why it matters
Regulatory assistants often fail by producing unsupported claims or non-compliant simplifications. SCRA treats quality as a gated process instead of trusting a single model pass. It targets teams that need explainable outputs and predictable runtime controls.

## What I built
- A Clean Architecture codebase with explicit domain, application, infrastructure, and presentation boundaries.
- Protocol-based ports/adapters so LLM, reranker, retriever, and web-search providers can be swapped cleanly.
- A LangGraph state machine for retrieve -> generate -> grade -> refine/fallback routing with retry budgeting.
- A FastAPI API surface with `/query`, `/health`, `/ingest`, `/ingest/{task_id}`, and `/stats`.
- A retrieval pipeline using ChromaDB retrieval, Cohere reranking, sibling expansion, and source-priority boosting.
- LLM adapters for Groq/OpenAI with structured output, retries, token tracking, and audit hooks.
- EU AI Act ingestion with article/recital/annex parsing and resilient fallback extraction.
- Runtime middleware for per-IP rate limiting and hard latency budgets.
- Evaluation tooling for end-to-end QA metrics and retrieval chunking benchmarks, wired into CI checks.

## How it works (high level)
Input starts as a user question submitted to `/query`. The API checks cache and initializes graph state.

Retrieval pulls top-k chunks from ChromaDB, reranks with Cohere, and may expand sibling chunks for local context. If fallback is active, Tavily web search adds evidence.

Generation returns answer text plus cited source IDs. Grading runs grounding and compliance checks concurrently, then citation validation confirms cited IDs exist in the evidence set.

Routing depends on grounding, compliance, evidence sufficiency, quality improvement, and retry budget. The system can refine, activate fallback retrieval, or return the best attempt when budgets are exhausted. Output includes scores, sources, latency, fallback flag, and legal disclaimer.

## AI & data approach
SCRA uses LLMs in distinct roles: generation, grounding grading, and compliance grading. Structured outputs are preferred, with controlled parse fallbacks when structured calls fail.

Retrieval uses dense search plus reranking before generation. Chunking is configurable (`fixed` or `paragraph`) with overlap, and benchmark artifacts compare quality across chunk sizes and metadata strategies. Source IDs are regex-validated so citations stay machine-checkable across ingestion and response validation.

Prompting is role-specific across generation, grounding, compliance, refinement, and query rewriting. Refinement prompts include grading feedback so retries are corrective. Evidence blocks are truncated by character budget to control prompt size.

Data handling is traceability-first: chunk metadata carries section type and source URL, and citation checks enforce source consistency. Dedicated PII redaction is not implemented yet.

## Engineering choices (production mindset)
- Reliability: LLM, reranker, and web-search adapters use bounded retries with exponential backoff.
- Reliability: workflow routing includes stall detection, evidence-insufficiency handling, retry limits, and best-answer fallback.
- Observability: structured logging, OpenTelemetry tracing, optional OTLP export, and optional Langfuse callback integration.
- Performance control: latency middleware enforces hard request budgets and returns explicit 504 responses when exceeded.
- Capacity control: per-IP token-bucket rate limiting plus query caching to reduce repeated full pipeline runs.
- Cost visibility: `/stats` reports call counts and token usage to support monitoring and budget decisions.
- Quality enforcement: strict source ID patterns, citation validation, and evidence truncation rules reduce malformed or unsupported outputs.
- Evaluation discipline: committed eval harness, retrieval benchmarks, and CI gates with lint, tests, and coverage threshold.
- Not implemented yet: distributed cache and shared ingestion-task state for multi-worker deployments.

## Tradeoffs & design decisions
- Optimized for grounded, auditable answers over raw speed; grading and retries increase latency tails.
- Optimized for modular provider abstraction; cleaner boundaries add implementation complexity.
- Chose in-process cache/task state for simplicity; easier local operation but weaker horizontal scale.
- Chose external reranking and web fallback for quality recovery; better coverage but more network dependency.

## Results (if available)
Measured results are available in committed evaluation artifacts.

End-to-end results from February 11, 2026 show two labeled runs. The `gpt-4o-mini` run over 25 questions reports pass rate 72%, mean relevance 0.69, grounding 96%, compliance 96%, citation validity 88%, citation-source match 56%, fallback 20%, and mean latency about 22.8s. The `maverick`-labeled run over 10 questions reports pass rate 70%, grounding 100%, compliance 100%, citation validity 90%, citation-source match 70%, fallback 0%, and mean latency about 58.4s.

Retrieval benchmark results from February 17, 2026 rank `fixed_2000` and `para_2000` tied for top composite score, with recall@5 about 41.8%, recall@10 about 61.6%, recall@25 about 82.6%, and hit@1 at 100%. Different sample sizes mean these results are directional, not directly comparable across runs.

## FAQ (for my resume chatbot)
**Q: What problem does this solve?**  
A: SCRA answers EU AI Act questions with explicit source grounding and compliance checks. It reduces unsupported legal claims and provides a transparent citation trail.

**Q: What is the architecture?**  
A: The system uses Clean Architecture with domain, application, infrastructure, and presentation layers. A LangGraph workflow coordinates retrieval, generation, grading, and fallback decisions.

**Q: Why did you choose RAG/agents?**  
A: RAG is necessary because regulatory answers must be tied to source text, not model memory. Agent-style routing decides when to refine, retry, or escalate to web fallback.

**Q: How do you handle hallucinations/reliability?**  
A: Every answer is graded for grounding and compliance, and citations are validated against retrieved evidence IDs. Weak outputs trigger refinement or fallback paths under bounded retry and latency budgets.

**Q: How do you evaluate quality?**  
A: The project includes a golden-dataset harness with relevance, grounding, compliance, citation, fallback, and latency metrics. It also includes retrieval-only benchmarking for chunking strategy decisions.

**Q: What was the hardest part?**  
A: The hardest part was balancing trust signals with response time. Multi-step quality control improves reliability but increases tail latency when retries or fallback activate.

**Q: What would you improve next?**  
A: The next upgrade is shared state for cache and ingestion tracking so the service scales across workers. After that, stronger output guardrails and embedding/chunking experiments would improve robustness.

**Q: What is your contribution?**  
A: `my_role` is marked unknown because explicit ownership metadata is not documented. The implementation covers architecture, orchestration, adapters, middleware, evaluation tooling, and CI.

## Fact Sheet (for precise retrieval)
- Primary user: Compliance, legal operations, and policy users working on EU AI Act obligations.
- Inputs: Natural-language regulatory questions, indexed EU AI Act text, optional fallback web evidence.
- Outputs: Cited answer, grounding score, compliance analysis, source IDs, latency, fallback flag.
- Core AI pattern: RAG with self-correction, grading gates, and tool-based fallback.
- Storage: ChromaDB persistent vector store plus in-memory cache and ingestion task state.
- Interfaces (API/UI/CLI): FastAPI HTTP API and CLI eval/benchmark scripts; no dedicated UI.
- Deployment: Local uvicorn service or Docker/docker-compose with persisted Chroma volume.
- Key risks + mitigations (if any): Hallucinations mitigated by grading+citation checks; latency spikes mitigated by timeout budgets and cache; abuse mitigated by per-IP token bucket.
- Current status: Unknown in metadata; code and eval artifacts indicate active prototype iteration in February 2026.
