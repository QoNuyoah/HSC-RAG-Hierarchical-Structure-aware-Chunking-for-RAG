# -*- coding: utf-8 -*-
"""BM25 retriever for chunk-level RAG evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - local fallback for portability
    BM25Okapi = None


TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)
CJK_SPAN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?")
TokenizerProfile = Literal["mixed", "cjk_bigram", "cjk_2_4gram", "jieba"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

CHINESE_STOPWORDS = {
    "一个",
    "一种",
    "一些",
    "以及",
    "可以",
    "我们",
    "他们",
    "这个",
    "这些",
    "进行",
    "没有",
    "不是",
    "就是",
    "因为",
    "所以",
    "如果",
    "什么",
    "怎么",
    "如何",
    "哪些",
    "多少",
    "为什么",
}


def tokenize(text: str, profile: TokenizerProfile = "mixed") -> list[str]:
    """Tokenize English-heavy scientific text with light CJK support."""

    if profile == "mixed":
        return _tokenize_mixed(text)
    if profile == "cjk_bigram":
        return _tokenize_cjk_bigram(text)
    if profile == "cjk_2_4gram":
        return _tokenize_cjk_2_4gram(text)
    if profile == "jieba":
        return _tokenize_jieba(text)
    raise ValueError(f"Unknown tokenizer profile: {profile}")


def _tokenize_mixed(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    return [token for token in tokens if token not in STOPWORDS]


def _tokenize_cjk_bigram(text: str) -> list[str]:
    text = text or ""
    tokens = [token.lower() for token in LATIN_TOKEN_RE.findall(text)]
    for span in CJK_SPAN_RE.findall(text):
        tokens.extend(_cjk_ngrams(span, min_n=2, max_n=3))
    return _filter_tokens(tokens)


def _tokenize_cjk_2_4gram(text: str) -> list[str]:
    text = text or ""
    tokens = [token.lower() for token in LATIN_TOKEN_RE.findall(text)]
    for span in CJK_SPAN_RE.findall(text):
        tokens.extend(_cjk_ngrams(span, min_n=2, max_n=4))
    return _filter_tokens(tokens)


def _tokenize_jieba(text: str) -> list[str]:
    text = text or ""
    try:
        import jieba
    except ImportError:  # pragma: no cover - depends on optional runtime package.
        return _tokenize_cjk_bigram(text)

    tokens: list[str] = []
    for item in jieba.lcut(text, cut_all=False):
        item = item.strip().lower()
        if not item:
            continue
        if LATIN_TOKEN_RE.fullmatch(item):
            tokens.append(item)
            continue
        if CJK_SPAN_RE.fullmatch(item):
            if len(item) >= 2:
                tokens.append(item)
            # Add char n-grams for noisy CJK text and short natural-language queries.
            tokens.extend(_cjk_ngrams(item, min_n=2, max_n=3))
    return _filter_tokens(tokens)


def _cjk_ngrams(text: str, *, min_n: int, max_n: int) -> list[str]:
    if not text:
        return []
    if len(text) < min_n:
        return [text]
    result: list[str] = []
    for n in range(min_n, min(max_n, len(text)) + 1):
        result.extend(text[index : index + n] for index in range(0, len(text) - n + 1))
    return result


def _filter_tokens(tokens: list[str]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token in STOPWORDS or token in CHINESE_STOPWORDS:
            continue
        if len(token) == 1 and not CJK_SPAN_RE.fullmatch(token):
            continue
        result.append(token)
    return result


class _FallbackBM25:
    """Small BM25Okapi-compatible fallback used only if rank_bm25 is missing."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        document_frequency: Counter[str] = Counter()
        for doc in corpus:
            for token in set(doc):
                document_frequency[token] += 1
        total_docs = len(corpus)
        self.idf = {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for freqs, doc_len in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            length_norm = 1 - self.b + self.b * (doc_len / self.avgdl) if self.avgdl else 1.0
            for token in query_tokens:
                term_freq = freqs.get(token, 0)
                if not term_freq:
                    continue
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * length_norm
                score += self.idf.get(token, 0.0) * numerator / denominator
            scores.append(score)
        return scores


@dataclass(frozen=True)
class Bm25Hit:
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


class BM25ChunkRetriever:
    """BM25 retriever over generated RAG chunks."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        include_metadata: bool = False,
        tokenizer_profile: TokenizerProfile = "mixed",
    ):
        self.chunks = chunks
        self.include_metadata = include_metadata
        self.tokenizer_profile = tokenizer_profile
        corpus = [self._tokens_for_chunk(chunk) for chunk in chunks]
        corpus = [tokens if tokens else ["__empty__"] for tokens in corpus]
        if BM25Okapi is not None:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = _FallbackBM25(corpus)

    def candidate_chunks(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        if doc_id is None:
            return self.chunks
        return [chunk for chunk in self.chunks if chunk.get("doc_id") == doc_id]

    def search(self, query: str, top_k: int = 5, doc_id: str | None = None) -> list[Bm25Hit]:
        query_tokens = tokenize(query, profile=self.tokenizer_profile)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda idx: (-float(scores[idx]), idx),
        )
        hits: list[Bm25Hit] = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if doc_id is not None and chunk.get("doc_id") != doc_id:
                continue
            hits.append(self._hit_from_chunk(chunk, rank=len(hits) + 1, score=float(scores[idx])))
            if len(hits) >= top_k:
                break
        return hits

    def _tokens_for_chunk(self, chunk: dict[str, Any]) -> list[str]:
        parts = [chunk.get("text") or ""]
        if self.include_metadata:
            parts.extend(chunk.get("title_path") or [])
            parts.extend(chunk.get("tags") or [])
            summary = chunk.get("summary")
            if summary:
                parts.append(summary)
        return tokenize(" ".join(str(part) for part in parts), profile=self.tokenizer_profile)

    def config(self) -> dict[str, Any]:
        return {
            "include_metadata": self.include_metadata,
            "tokenizer_profile": self.tokenizer_profile,
        }

    def _hit_from_chunk(self, chunk: dict[str, Any], rank: int, score: float) -> Bm25Hit:
        text = " ".join((chunk.get("text") or "").split())
        return Bm25Hit(
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
