# -*- coding: utf-8 -*-
"""Dense FAISS retriever for chunk-level RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class DenseHit:
    rank: int
    chunk_id: str
    doc_id: str
    score: float
    source_blocks: list[str]
    title_path: list[str]
    token_count: int
    quality_flags: list[str]
    preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "source_blocks": self.source_blocks,
            "title_path": self.title_path,
            "token_count": self.token_count,
            "quality_flags": self.quality_flags,
            "preview": self.preview,
        }


class DenseFaissRetriever:
    """Dense-vector retriever backed by FAISS.

    `encoder="sentence_transformer"` uses a local SentenceTransformer model.
    `encoder="tfidf_svd"` is a deterministic local dense fallback: TF-IDF is
    compressed into dense latent vectors with SVD and indexed by FAISS.
    `encoder="auto"` tries SentenceTransformer first and falls back to TF-IDF+SVD.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        encoder: str = "tfidf_svd",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        include_metadata: bool = False,
        svd_dim: int = 128,
        local_files_only: bool = True,
    ):
        self.chunks = chunks
        self.encoder_request = encoder
        self.model_name = model_name
        self.include_metadata = include_metadata
        self.svd_dim = svd_dim
        self.local_files_only = local_files_only
        self._texts = [self._text_for_chunk(chunk) for chunk in chunks]
        self._model = None
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self.encoder_name = ""
        self.encoder_warning: str | None = None

        vectors = self._encode_corpus()
        vectors = self._as_float32(vectors)
        self.dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 1
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def candidate_chunks(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        if doc_id is None:
            return self.chunks
        return [chunk for chunk in self.chunks if chunk.get("doc_id") == doc_id]

    def search(self, query: str, top_k: int = 5, doc_id: str | None = None) -> list[DenseHit]:
        if not self.chunks:
            return []
        query_vector = self._as_float32(self._encode_queries([query]))
        scores, indices = self.index.search(query_vector, len(self.chunks))
        hits: list[DenseHit] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if doc_id is not None and chunk.get("doc_id") != doc_id:
                continue
            hits.append(self._hit_from_chunk(chunk, rank=len(hits) + 1, score=float(score)))
            if len(hits) >= top_k:
                break
        return hits

    def config(self) -> dict[str, Any]:
        return {
            "encoder_request": self.encoder_request,
            "encoder_name": self.encoder_name,
            "model_name": self.model_name if self._model is not None else None,
            "include_metadata": self.include_metadata,
            "svd_dim": self.svd_dim,
            "local_files_only": self.local_files_only,
            "warning": self.encoder_warning,
            "faiss_index": "IndexFlatIP",
            "embedding_dim": self.dim,
        }

    def _encode_corpus(self) -> np.ndarray:
        if self.encoder_request in {"auto", "sentence_transformer"}:
            try:
                return self._encode_corpus_sentence_transformer()
            except Exception as exc:  # pragma: no cover - depends on local model cache
                if self.encoder_request == "sentence_transformer":
                    raise
                self.encoder_warning = (
                    f"SentenceTransformer unavailable locally; fell back to tfidf_svd. "
                    f"Reason: {type(exc).__name__}: {exc}"
                )
        return self._encode_corpus_tfidf_svd()

    def _encode_corpus_sentence_transformer(self) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(
                self.model_name,
                device="cpu",
                local_files_only=self.local_files_only,
            )
        except TypeError:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        vectors = self._model.encode(
            self._texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.encoder_name = f"sentence_transformer:{self.model_name}"
        return np.asarray(vectors, dtype=np.float32)

    def _encode_corpus_tfidf_svd(self) -> np.ndarray:
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=8000,
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_\-']+\b",
        )
        matrix = self._vectorizer.fit_transform(self._texts)
        if matrix.shape[0] <= 2 or matrix.shape[1] <= 2:
            dense = matrix.toarray()
            self._svd = None
        else:
            components = max(1, min(self.svd_dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
            self._svd = TruncatedSVD(n_components=components, random_state=42)
            dense = self._svd.fit_transform(matrix)
        vectors = normalize(np.asarray(dense, dtype=np.float32), norm="l2")
        self.encoder_name = "tfidf_svd"
        return vectors

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        if self._model is not None:
            vectors = self._model.encode(
                queries,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)
        if self._vectorizer is None:
            raise RuntimeError("Dense retriever is not initialized")
        matrix = self._vectorizer.transform(queries)
        if self._svd is not None:
            dense = self._svd.transform(matrix)
        else:
            dense = matrix.toarray()
        return normalize(np.asarray(dense, dtype=np.float32), norm="l2")

    def _as_float32(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.size == 0:
            vectors = np.zeros((len(self.chunks), 1), dtype=np.float32)
        return np.ascontiguousarray(vectors)

    def _text_for_chunk(self, chunk: dict[str, Any]) -> str:
        parts = [chunk.get("text") or ""]
        if self.include_metadata:
            parts.extend(chunk.get("title_path") or [])
            parts.extend(chunk.get("tags") or [])
            summary = chunk.get("summary")
            if summary:
                parts.append(summary)
        return " ".join(str(part) for part in parts)

    def _hit_from_chunk(self, chunk: dict[str, Any], rank: int, score: float) -> DenseHit:
        text = " ".join((chunk.get("text") or "").split())
        return DenseHit(
            rank=rank,
            chunk_id=str(chunk.get("chunk_id")),
            doc_id=str(chunk.get("doc_id")),
            score=round(score, 6),
            source_blocks=list(chunk.get("source_blocks") or []),
            title_path=list(chunk.get("title_path") or []),
            token_count=int(chunk.get("token_count") or 0),
            quality_flags=list(chunk.get("quality_flags") or []),
            preview=text[:240],
        )

