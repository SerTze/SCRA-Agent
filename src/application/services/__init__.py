"""Application services – shared, single-responsibility classes.

These services contain ALL business logic so that nodes and API handlers
remain thin orchestrators. No logic is duplicated across nodes.

Modules:
    evidence_builder  – Shared evidence block builder for LLM prompts.
    citation_service  – Citation validation and enforcement.
    generation_service – RAG prompt building and answer generation.
    grading_service   – Grounding and compliance grading via LLM.
    retriever_service – Retrieval, reranking, sibling expansion, boosting.
"""
