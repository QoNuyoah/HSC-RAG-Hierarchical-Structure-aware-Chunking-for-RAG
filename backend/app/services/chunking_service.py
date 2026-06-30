# -*- coding: utf-8 -*-
"""Shared chunking service for standard APIs and agent orchestration."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields
from typing import Any

from fastapi import HTTPException

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


CHUNKER_REGISTRY = {
    "fixed": (FixedSizeChunker, FixedChunkConfig),
    "recursive": (RecursiveChunker, RecursiveChunkConfig),
    "semantic": (SemanticChunker, SemanticChunkConfig),
    "hsc_rag": (HscRagChunker, HscRagConfig),
}


def build_chunker(strategy: ChunkStrategy, config: dict[str, Any]):
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


def build_chunk_report(doc: GovernedDocument, chunks: list[RagChunk], config: dict[str, Any]) -> dict[str, Any]:
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
        "boundary_score_summary": boundary_score_summary(chunks),
    }


def boundary_score_summary(chunks: list[RagChunk]) -> dict[str, Any]:
    decisions = [
        decision
        for chunk in chunks
        if isinstance((decision := (chunk.metadata or {}).get("closing_boundary_decision")), dict)
    ]
    if not decisions:
        return {}

    scores = [float(item.get("boundary_score", 0.0)) for item in decisions]
    distances = [
        float((item.get("signals") or {}).get("semantic_distance", 0.0))
        for item in decisions
    ]
    similarities = [
        float((item.get("signals") or {}).get("semantic_similarity", 0.0))
        for item in decisions
    ]
    reasons = Counter(str(item.get("split_reason", "unknown")) for item in decisions)
    semantic_triggered = sum(
        1
        for item in decisions
        if (item.get("signals") or {}).get("semantic_boundary_triggered")
    )
    return {
        "scored_boundaries": len(decisions),
        "chunk_coverage_rate": round(len(decisions) / len(chunks), 4) if chunks else 0.0,
        "avg_boundary_score": round(sum(scores) / len(scores), 4),
        "max_boundary_score": round(max(scores), 4),
        "avg_semantic_distance": round(sum(distances) / len(distances), 4),
        "avg_semantic_similarity": round(sum(similarities) / len(similarities), 4),
        "semantic_boundary_triggered": semantic_triggered,
        "split_reason_counts": dict(sorted(reasons.items())),
    }


def run_chunk_request(request: ChunkAgentRequest) -> ChunkAgentResponse:
    chunker = build_chunker(request.strategy, request.config)
    chunks = chunker.chunk_document(request.document)
    return ChunkAgentResponse(
        strategy=request.strategy,
        doc_id=request.document.doc_id,
        chunks=chunks,
        chunk_count=len(chunks),
        report=build_chunk_report(request.document, chunks, request.config) if request.include_report else {},
    )


def run_chunk_batch_request(request: ChunkBatchAgentRequest) -> ChunkBatchAgentResponse:
    results = [
        run_chunk_request(
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
