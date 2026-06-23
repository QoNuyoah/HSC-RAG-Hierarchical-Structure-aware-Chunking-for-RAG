# -*- coding: utf-8 -*-
"""Recursive text-splitting baseline.

This approximates a common RecursiveCharacterTextSplitter-style baseline while
preserving GovernedBlock provenance for fair evidence-retrieval evaluation.
It is intentionally structure-agnostic: section hierarchy is not used as a
primary boundary signal.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunkers.common import (
    block_text,
    content_blocks,
    estimate_tokens,
    make_chunk,
    normalize_text,
    word_chunks,
)
from app.core.schemas import GovernedBlock, GovernedDocument, RagChunk


@dataclass(frozen=True)
class RecursiveChunkConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 128
    max_tokens: int = 512
    include_title_context: bool = False
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class _Piece:
    text: str
    block: GovernedBlock
    split_level: int


class RecursiveChunker:
    strategy = "recursive"

    def __init__(self, config: RecursiveChunkConfig | None = None):
        self.config = config or RecursiveChunkConfig()

    def chunk_document(self, doc: GovernedDocument) -> list[RagChunk]:
        pieces: list[_Piece] = []
        for block in content_blocks(doc):
            text = block_text(block, include_title_path=self.config.include_title_context)
            text = normalize_text(text)
            if not text:
                continue
            pieces.extend(self._split_text(text, block, separator_index=0))

        chunks: list[RagChunk] = []
        buffer: list[_Piece] = []
        buffer_tokens = 0
        index = 1

        def flush() -> None:
            nonlocal buffer, buffer_tokens, index
            if not buffer:
                return
            text = " ".join(piece.text for piece in buffer).strip()
            blocks = self._unique_blocks([piece.block for piece in buffer])
            split_levels = sorted({piece.split_level for piece in buffer})
            chunks.append(
                make_chunk(
                    doc=doc,
                    strategy=self.strategy,
                    index=index,
                    text=text,
                    blocks=blocks,
                    min_tokens=self.config.min_tokens,
                    max_tokens=self.config.max_tokens,
                    extra_flags=["recursive_baseline"],
                    metadata={
                        "target_tokens": self.config.target_tokens,
                        "overlap_tokens": self.config.overlap_tokens,
                        "include_title_context": self.config.include_title_context,
                        "split_levels": split_levels,
                    },
                )
            )
            index += 1

            overlap: list[_Piece] = []
            overlap_tokens = 0
            for piece in reversed(buffer):
                piece_tokens = estimate_tokens(piece.text)
                if overlap and overlap_tokens + piece_tokens > self.config.overlap_tokens:
                    break
                if piece_tokens > self.config.overlap_tokens:
                    break
                overlap.append(piece)
                overlap_tokens += piece_tokens
                if overlap_tokens >= self.config.overlap_tokens:
                    break
            overlap.reverse()
            if len(overlap) == len(buffer):
                overlap = []
                overlap_tokens = 0
            buffer = overlap
            buffer_tokens = overlap_tokens

        for piece in pieces:
            piece_tokens = estimate_tokens(piece.text)
            if not piece_tokens:
                continue
            if buffer and buffer_tokens + piece_tokens > self.config.target_tokens:
                flush()
            buffer.append(piece)
            buffer_tokens += piece_tokens
            if buffer_tokens >= self.config.max_tokens:
                flush()
        flush()
        return chunks

    def _split_text(
        self,
        text: str,
        block: GovernedBlock,
        separator_index: int,
    ) -> list[_Piece]:
        text = normalize_text(text)
        if not text:
            return []
        if estimate_tokens(text) <= self.config.max_tokens:
            return [_Piece(text=text, block=block, split_level=separator_index)]

        if separator_index >= len(self.config.separators):
            return [
                _Piece(text=part, block=block, split_level=separator_index)
                for part in word_chunks(text, max_tokens=self.config.max_tokens, overlap_tokens=0)
            ]

        separator = self.config.separators[separator_index]
        parts = self._split_by_separator(text, separator)
        if len(parts) <= 1:
            return self._split_text(text, block, separator_index + 1)

        pieces: list[_Piece] = []
        buffer: list[str] = []
        buffer_tokens = 0
        for part in parts:
            part = normalize_text(part)
            if not part:
                continue
            part_tokens = estimate_tokens(part)
            if part_tokens > self.config.max_tokens:
                if buffer:
                    pieces.extend(
                        self._split_text(" ".join(buffer), block, separator_index + 1)
                    )
                    buffer = []
                    buffer_tokens = 0
                pieces.extend(self._split_text(part, block, separator_index + 1))
                continue
            if buffer and buffer_tokens + part_tokens > self.config.max_tokens:
                pieces.append(
                    _Piece(
                        text=" ".join(buffer),
                        block=block,
                        split_level=separator_index,
                    )
                )
                buffer = [part]
                buffer_tokens = part_tokens
            else:
                buffer.append(part)
                buffer_tokens += part_tokens
        if buffer:
            pieces.append(
                _Piece(
                    text=" ".join(buffer),
                    block=block,
                    split_level=separator_index,
                )
            )
        return pieces

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        if separator == " ":
            return text.split()
        if separator in {". ", "; ", ", "}:
            parts = text.split(separator)
            return [part + separator.strip() if idx < len(parts) - 1 else part for idx, part in enumerate(parts)]
        return text.split(separator)

    def _unique_blocks(self, blocks: list[GovernedBlock]) -> list[GovernedBlock]:
        seen: set[str] = set()
        unique: list[GovernedBlock] = []
        for block in blocks:
            if block.block_id not in seen:
                seen.add(block.block_id)
                unique.append(block)
        return unique

