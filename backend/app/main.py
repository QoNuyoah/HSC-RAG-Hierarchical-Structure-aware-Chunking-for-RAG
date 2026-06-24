# -*- coding: utf-8 -*-
"""FastAPI application for the HSC-RAG chunking agent and demo dashboard."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.chunkers.fixed import FixedChunkConfig, FixedSizeChunker
from app.chunkers.hsc_rag import HscRagChunker, HscRagConfig
from app.chunkers.recursive import RecursiveChunkConfig, RecursiveChunker
from app.chunkers.semantic import SemanticChunkConfig, SemanticChunker
from app.core.schemas import (
    ChunkAgentRequest,
    ChunkAgentResponse,
    ChunkBatchAgentRequest,
    ChunkBatchAgentResponse,
    ChunkStrategy,
    GovernedDocument,
    RagChunk,
)
from app.services.evaluation_store import RETRIEVERS, EvaluationStore


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


CHUNKER_REGISTRY = {
    "fixed": (FixedSizeChunker, FixedChunkConfig),
    "recursive": (RecursiveChunker, RecursiveChunkConfig),
    "semantic": (SemanticChunker, SemanticChunkConfig),
    "hsc_rag": (HscRagChunker, HscRagConfig),
}


def _check_retriever(retriever: str) -> None:
    if retriever not in RETRIEVERS:
        raise HTTPException(status_code=400, detail=f"Unknown retriever: {retriever}")


def _build_chunker(strategy: ChunkStrategy, config: dict[str, Any]):
    chunker_cls, config_cls = CHUNKER_REGISTRY[strategy]
    allowed = {field.name for field in fields(config_cls)}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported config fields for strategy {strategy}",
                "unsupported_fields": unknown,
                "allowed_fields": sorted(allowed),
            },
        )
    return chunker_cls(config_cls(**config))


def _chunk_report(doc: GovernedDocument, chunks: list[RagChunk], config: dict[str, Any]) -> dict[str, Any]:
    token_counts = [chunk.token_count for chunk in chunks]
    quality_flags = Counter(flag for chunk in chunks for flag in chunk.quality_flags)
    return {
        "schema_version": "hsc-agent-api-v1",
        "input_contract": "GovernedDocument",
        "output_contract": "RagChunk[]",
        "governance_stage": doc.governance_stage,
        "normalization_status": doc.normalization_status,
        "config": config,
        "chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "avg_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else None,
        "min_tokens": min(token_counts) if token_counts else None,
        "max_tokens": max(token_counts) if token_counts else None,
        "quality_flag_counts": dict(sorted(quality_flags.items())),
    }


def _run_chunk_agent(request: ChunkAgentRequest) -> ChunkAgentResponse:
    chunker = _build_chunker(request.strategy, request.config)
    chunks = chunker.chunk_document(request.document)
    return ChunkAgentResponse(
        strategy=request.strategy,
        doc_id=request.document.doc_id,
        chunks=chunks,
        chunk_count=len(chunks),
        report=_chunk_report(request.document, chunks, request.config) if request.include_report else {},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hsc-rag-api"}


@app.post("/api/v1/chunk", response_model=ChunkAgentResponse)
def chunk_document(request: ChunkAgentRequest) -> ChunkAgentResponse:
    """Chunk one governed document and return a RAG-ready chunk sequence."""

    return _run_chunk_agent(request)


@app.post("/api/v1/chunk/batch", response_model=ChunkBatchAgentResponse)
def chunk_documents(request: ChunkBatchAgentRequest) -> ChunkBatchAgentResponse:
    """Chunk multiple governed documents for upstream conversion pipelines."""

    results = [
        _run_chunk_agent(
            ChunkAgentRequest(
                document=document,
                strategy=request.strategy,
                config=request.config,
                include_report=request.include_report,
            )
        )
        for document in request.documents
    ]
    return ChunkBatchAgentResponse(
        strategy=request.strategy,
        document_count=len(request.documents),
        total_chunks=sum(result.chunk_count for result in results),
        results=results,
    )


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
