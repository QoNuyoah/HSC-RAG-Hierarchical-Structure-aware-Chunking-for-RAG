# -*- coding: utf-8 -*-
"""FastAPI application for the HSC-RAG chunking agent and demo dashboard."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.core.schemas import (
    ChunkAgentRequest,
    ChunkAgentResponse,
    ChunkBatchAgentRequest,
    ChunkBatchAgentResponse,
    LangChainAgentRequest,
    LangChainAgentResponse,
)
from app.services.chunking_service import run_chunk_batch_request, run_chunk_request
from app.services.evaluation_store import RETRIEVERS, EvaluationStore
from app.services.langchain_agent_service import run_langchain_agent


app = FastAPI(
    title="HSC-RAG API",
    version="0.1.0",
    description=(
        "Standard chunking-agent API for HSC-RAG plus experiment APIs for "
        "retrieval evaluation and dashboard visualization."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = EvaluationStore()


def _check_retriever(retriever: str) -> None:
    if retriever not in RETRIEVERS:
        raise HTTPException(status_code=400, detail=f"Unknown retriever: {retriever}")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hsc-rag-api"}


@app.post("/api/v1/chunk", response_model=ChunkAgentResponse)
def chunk_document(request: ChunkAgentRequest) -> ChunkAgentResponse:
    """Chunk one governed document and return a RAG-ready chunk sequence."""

    return run_chunk_request(request)


@app.post("/api/v1/chunk/batch", response_model=ChunkBatchAgentResponse)
def chunk_documents(request: ChunkBatchAgentRequest) -> ChunkBatchAgentResponse:
    """Chunk multiple governed documents for upstream conversion pipelines."""

    return run_chunk_batch_request(request)


@app.post("/api/v1/agent/run", response_model=LangChainAgentResponse)
def run_agent(request: LangChainAgentRequest) -> LangChainAgentResponse:
    """Run the LangChain-backed HSC-RAG agent over governed documents."""

    return run_langchain_agent(request)


@app.get("/api/overview")
def overview() -> dict:
    return store.overview()


@app.get("/api/metrics")
def metrics(retriever: str | None = Query(default=None)) -> dict:
    if retriever is not None:
        _check_retriever(retriever)
    return store.metrics(retriever)


@app.get("/api/queries")
def queries(retriever: str = Query(default="bm25")) -> dict:
    _check_retriever(retriever)
    return store.queries(retriever)


@app.get("/api/bad-cases")
def bad_cases(retriever: str = Query(default="bm25")) -> dict:
    _check_retriever(retriever)
    return store.bad_cases(retriever)


@app.get("/api/queries/{query_id}/comparison")
def query_comparison(query_id: str, retriever: str = Query(default="bm25")) -> dict:
    _check_retriever(retriever)
    result = store.query_comparison(query_id, retriever)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=f"Query not found: {query_id}")
    return result


@app.post("/api/cache/refresh")
def refresh_cache() -> dict[str, str]:
    store.clear_cache()
    return {"status": "refreshed"}
