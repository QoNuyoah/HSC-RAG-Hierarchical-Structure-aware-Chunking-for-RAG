# -*- coding: utf-8 -*-
"""Fixed-size chunking baseline.

This is intentionally simple and serves as the mechanical baseline for
evaluating HSC-RAG. It preserves source block metadata for fair retrieval
evaluation, but it does not optimize for structure boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.chunkers.common import (
    block_text,
    content_blocks,
    estimate_tokens,
    make_chunk,
    word_chunks,
)
from app.core.schemas import GovernedBlock, GovernedDocument, RagChunk


@dataclass(frozen=True)
class FixedChunkConfig:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_tokens: int = 128
    max_tokens: int = 512
    include_title_context: bool = False


class FixedSizeChunker:
    strategy = "fixed"

    def __init__(self, config: FixedChunkConfig | None = None):
        self.config = config or FixedChunkConfig()

    def chunk_document(self, doc: GovernedDocument) -> list[RagChunk]:
        blocks = content_blocks(doc)
        chunks: list[RagChunk] = []
        buffer_texts: list[str] = []
        buffer_blocks: list[GovernedBlock] = []
        buffer_tokens = 0
        index = 1

        def flush(extra_flags: list[str] | None = None) -> None:
            nonlocal buffer_texts, buffer_blocks, buffer_tokens, index
            text = "\n\n".join(buffer_texts).strip()
            if not text or not buffer_blocks:
                buffer_texts = []
                buffer_blocks = []
                buffer_tokens = 0
                return
            chunks.append(
                make_chunk(
                    doc=doc,
                    strategy=self.strategy,
                    index=index,
                    text=text,
                    blocks=buffer_blocks,
                    min_tokens=self.config.min_tokens,
                    max_tokens=self.config.max_tokens,
                    extra_flags=(extra_flags or []) + ["fixed_window_baseline"],
                    metadata={
                        "target_tokens": self.config.target_tokens,
                        "overlap_tokens": self.config.overlap_tokens,
                        "include_title_context": self.config.include_title_context,
                    },
                )
            )
            index += 1
            buffer_texts = []
            buffer_blocks = []
            buffer_tokens = 0

        for block in blocks:
            text = block_text(block, include_title_path=self.config.include_title_context)
            block_tokens = estimate_tokens(text)
            if not text:
                continue

            if block_tokens > self.config.target_tokens:
                flush()
                parts = word_chunks(
                    text,
                    max_tokens=self.config.target_tokens,
                    overlap_tokens=self.config.overlap_tokens,
                )
                for part in parts:
                    chunks.append(
                        make_chunk(
                            doc=doc,
                            strategy=self.strategy,
                            index=index,
                            text=part,
                            blocks=[block],
                            min_tokens=self.config.min_tokens,
                            max_tokens=self.config.max_tokens,
                            extra_flags=["fixed_window_baseline", "split_long_block"],
                            metadata={
                                "target_tokens": self.config.target_tokens,
                                "overlap_tokens": self.config.overlap_tokens,
                                "source_block_token_estimate": block_tokens,
                            },
                        )
                    )
                    index += 1
                continue

            if buffer_texts and buffer_tokens + block_tokens > self.config.target_tokens:
                flush()

            buffer_texts.append(text)
            buffer_blocks.append(block)
            buffer_tokens += block_tokens

        flush()
        return chunks

