# -*- coding: utf-8 -*-
"""Hybrid BM25 + dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.retrievers.bm25 import BM25ChunkRetriever
from app.retrievers.dense_faiss import DenseFaissRetriever


@dataclass(frozen=True)
class HybridHit:
    rank: int
    chunk_id: str
    doc_id: str
    score: float
    source_blocks: list[str]
    title_path: list[str]
    token_count: int
    quality_flags: list[str]
    preview: str
    bm25_score: float
    dense_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "source_blocks": self.source_blocks,
            "title_path": self.title_path,
            "token_count": self.token_count,
            "quality_flags": self.quality_flags,
            "preview": self.preview,
        }


class HybridRetriever:
    """Weighted score fusion over BM25 and dense FAISS rankings."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        alpha: float = 0.55,
        include_metadata: bool = False,
        dense_encoder: str = "tfidf_svd",
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_svd_dim: int = 128,
        local_files_only: bool = True,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.chunks = chunks
        self.alpha = alpha
        self.bm25 = BM25ChunkRetriever(chunks, include_metadata=include_metadata)
        self.dense = DenseFaissRetriever(
            chunks,
            encoder=dense_encoder,
            model_name=dense_model_name,
            include_metadata=include_metadata,
            svd_dim=dense_svd_dim,
            local_files_only=local_files_only,
        )

    def candidate_chunks(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        if doc_id is None:
            return self.chunks
        return [chunk for chunk in self.chunks if chunk.get("doc_id") == doc_id]

    def search(self, query: str, top_k: int = 5, doc_id: str | None = None) -> list[HybridHit]:
        candidate_count = len(self.candidate_chunks(doc_id))
        bm25_hits = self.bm25.search(query, top_k=candidate_count, doc_id=doc_id)
        dense_hits = self.dense.search(query, top_k=candidate_count, doc_id=doc_id)
        bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        bm25_norm = self._normalize({hit.chunk_id: hit.score for hit in bm25_hits})
        dense_norm = self._normalize({hit.chunk_id: hit.score for hit in dense_hits})

        fused: list[tuple[float, str]] = []
        for chunk_id in set(bm25_by_id) | set(dense_by_id):
            score = self.alpha * bm25_norm.get(chunk_id, 0.0) + (1 - self.alpha) * dense_norm.get(chunk_id, 0.0)
            fused.append((score, chunk_id))
        fused.sort(key=lambda item: (-item[0], item[1]))

        hits: list[HybridHit] = []
        for score, chunk_id in fused[:top_k]:
            source_hit = bm25_by_id.get(chunk_id) or dense_by_id[chunk_id]
            hits.append(
                HybridHit(
                    rank=len(hits) + 1,
                    chunk_id=source_hit.chunk_id,
                    doc_id=source_hit.doc_id,
                    score=round(score, 6),
                    source_blocks=source_hit.source_blocks,
                    title_path=source_hit.title_path,
                    token_count=source_hit.token_count,
                    quality_flags=source_hit.quality_flags,
                    preview=source_hit.preview,
                    bm25_score=bm25_by_id.get(chunk_id).score if chunk_id in bm25_by_id else 0.0,
                    dense_score=dense_by_id.get(chunk_id).score if chunk_id in dense_by_id else 0.0,
                )
            )
        return hits

    def config(self) -> dict[str, Any]:
        return {
            "alpha_bm25": self.alpha,
            "dense": self.dense.config(),
        }

    def _normalize(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        low = min(values)
        high = max(values)
        if high == low:
            return {key: 0.0 for key in scores}
        return {key: (value - low) / (high - low) for key, value in scores.items()}

