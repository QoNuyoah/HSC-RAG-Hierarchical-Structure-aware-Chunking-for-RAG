# -*- coding: utf-8 -*-
"""FastAPI application for the HSC-RAG demo dashboard."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.services.evaluation_store import RETRIEVERS, EvaluationStore


app = FastAPI(
    title="HSC-RAG API",
    version="0.1.0",
    description="Experiment API for HSC-RAG chunking and retrieval evaluation.",
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

