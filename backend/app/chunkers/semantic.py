# -*- coding: utf-8 -*-
"""Semantic chunking baseline using local TF-IDF sentence cohesion."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.chunkers.common import (
    PROTECTED_BLOCK_TYPES,
    block_text,
    content_blocks,
    estimate_tokens,
    make_chunk,
    normalize_text,
    sentence_chunks,
)
from app.core.schemas import GovernedBlock, GovernedDocument, RagChunk


SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class SemanticChunkConfig:
    min_tokens: int = 160
    target_tokens: int = 512
    max_tokens: int = 768
    breakpoint_percentile: float = 75.0
    include_title_context: bool = False
    protect_blocks: bool = True


@dataclass(frozen=True)
class _SentenceUnit:
    text: str
    block: GovernedBlock


class SemanticChunker:
    """Baseline that uses local sentence-level semantic distance.

    Unlike HSC-RAG, this baseline does not privilege document hierarchy. It
    estimates adjacent sentence cohesion and starts a new chunk when semantic
    distance is high and the current chunk is already sufficiently large.
    """

    strategy = "semantic"

    def __init__(self, config: SemanticChunkConfig | None = None):
        self.config = config or SemanticChunkConfig()

    def chunk_document(self, doc: GovernedDocument) -> list[RagChunk]:
        units = self._sentence_units(doc)
        if not units:
            return []
        break_after = self._semantic_breakpoints(units)

        chunks: list[RagChunk] = []
        buffer: list[_SentenceUnit] = []
        buffer_tokens = 0
        index = 1

        def flush(extra_flags: list[str] | None = None) -> None:
            nonlocal buffer, buffer_tokens, index
            if not buffer:
                return
            text = " ".join(unit.text for unit in buffer)
            chunks.append(
                make_chunk(
                    doc=doc,
                    strategy=self.strategy,
                    index=index,
                    text=text,
                    blocks=self._unique_blocks([unit.block for unit in buffer]),
                    min_tokens=self.config.min_tokens,
                    max_tokens=self.config.max_tokens,
                    extra_flags=(extra_flags or []) + ["semantic_baseline"],
                    metadata={
                        "target_tokens": self.config.target_tokens,
                        "max_tokens": self.config.max_tokens,
                        "breakpoint_percentile": self.config.breakpoint_percentile,
                        "include_title_context": self.config.include_title_context,
                    },
                )
            )
            index += 1
            buffer = []
            buffer_tokens = 0

        for idx, unit in enumerate(units):
            unit_tokens = estimate_tokens(unit.text)
            if unit_tokens > self.config.max_tokens:
                flush()
                for part in sentence_chunks(unit.text, max_tokens=self.config.max_tokens):
                    chunks.append(
                        make_chunk(
                            doc=doc,
                            strategy=self.strategy,
                            index=index,
                            text=part,
                            blocks=[unit.block],
                            min_tokens=self.config.min_tokens,
                            max_tokens=self.config.max_tokens,
                            extra_flags=["semantic_baseline", "split_long_unit"],
                            metadata={
                                "target_tokens": self.config.target_tokens,
                                "source_block_token_estimate": unit_tokens,
                            },
                        )
                    )
                    index += 1
                continue

            if buffer and buffer_tokens + unit_tokens > self.config.max_tokens:
                flush(["max_length_boundary"])

            buffer.append(unit)
            buffer_tokens += unit_tokens

            should_break = idx in break_after and buffer_tokens >= self.config.min_tokens
            near_target = buffer_tokens >= self.config.target_tokens
            if should_break or near_target:
                flush(["semantic_boundary" if should_break else "target_length_boundary"])

        flush(["final_flush"])
        return chunks

    def _sentence_units(self, doc: GovernedDocument) -> list[_SentenceUnit]:
        units: list[_SentenceUnit] = []
        for block in content_blocks(doc):
            text = block_text(block, include_title_path=self.config.include_title_context)
            text = normalize_text(text)
            if not text:
                continue
            if self.config.protect_blocks and block.type in PROTECTED_BLOCK_TYPES:
                units.append(_SentenceUnit(text=text, block=block))
                continue
            sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
            if not sentences:
                sentences = [text]
            for sentence in sentences:
                units.append(_SentenceUnit(text=sentence, block=block))
        return units

    def _semantic_breakpoints(self, units: list[_SentenceUnit]) -> set[int]:
        if len(units) < 3:
            return set()
        texts = [unit.text for unit in units]
        try:
            matrix = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=5000,
                token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_\-']+\b",
            ).fit_transform(texts)
        except ValueError:
            return set()
        distances: list[float] = []
        for idx in range(len(units) - 1):
            sim = cosine_similarity(matrix[idx], matrix[idx + 1])[0][0]
            distances.append(1.0 - float(sim))
        if not distances:
            return set()
        threshold = float(np.percentile(np.array(distances), self.config.breakpoint_percentile))
        return {
            idx
            for idx, distance in enumerate(distances)
            if distance >= threshold and estimate_tokens(units[idx].text) > 0
        }

    def _unique_blocks(self, blocks: list[GovernedBlock]) -> list[GovernedBlock]:
        seen: set[str] = set()
        unique: list[GovernedBlock] = []
        for block in blocks:
            if block.block_id not in seen:
                seen.add(block.block_id)
                unique.append(block)
        return unique

