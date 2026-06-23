# -*- coding: utf-8 -*-
"""HSC-RAG: Hierarchical Structure-aware Chunking for RAG."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class HscRagConfig:
    min_tokens: int = 180
    target_tokens: int = 512
    max_tokens: int = 900
    include_title_context: bool = True
    merge_short_chunks: bool = True
    protect_blocks: bool = True


class HscRagChunker:
    """Structure-aware chunker aligned with the course requirement.

    It consumes governed content and focuses on result packaging:
    - preserve title hierarchy;
    - keep table/figure/code/formula/list blocks intact where possible;
    - control chunk length through merge/split logic;
    - emit traceable metadata for downstream RAG evaluation.
    """

    strategy = "hsc_rag"

    def __init__(self, config: HscRagConfig | None = None):
        self.config = config or HscRagConfig()

    def chunk_document(self, doc: GovernedDocument) -> list[RagChunk]:
        units = content_blocks(doc)
        initial = self._build_initial_chunks(doc, units)
        if self.config.merge_short_chunks:
            initial = self._merge_short_chunks(initial)
        return self._renumber(doc, initial)

    def _build_initial_chunks(
        self,
        doc: GovernedDocument,
        units: list[GovernedBlock],
    ) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        buffer: list[GovernedBlock] = []
        buffer_texts: list[str] = []
        buffer_tokens = 0
        index = 1

        def flush(extra_flags: list[str] | None = None) -> None:
            nonlocal buffer, buffer_texts, buffer_tokens, index
            if not buffer:
                buffer_texts = []
                buffer_tokens = 0
                return
            text = "\n\n".join(buffer_texts)
            chunks.append(
                make_chunk(
                    doc=doc,
                    strategy=self.strategy,
                    index=index,
                    text=text,
                    blocks=buffer,
                    min_tokens=self.config.min_tokens,
                    max_tokens=self.config.max_tokens,
                    extra_flags=(extra_flags or []) + ["hsc_structure_aware"],
                    metadata={
                        "target_tokens": self.config.target_tokens,
                        "max_tokens": self.config.max_tokens,
                        "include_title_context": self.config.include_title_context,
                    },
                )
            )
            index += 1
            buffer = []
            buffer_texts = []
            buffer_tokens = 0

        for unit in units:
            text = block_text(unit, include_title_path=self.config.include_title_context)
            text = normalize_text(text)
            if not text:
                continue
            unit_tokens = estimate_tokens(text)
            protected = unit.type in PROTECTED_BLOCK_TYPES
            same_section = self._same_title_path(buffer[-1], unit) if buffer else True

            if protected and self.config.protect_blocks and unit_tokens > self.config.max_tokens:
                flush()
                chunks.append(
                    make_chunk(
                        doc=doc,
                        strategy=self.strategy,
                        index=index,
                        text=text,
                        blocks=[unit],
                        min_tokens=self.config.min_tokens,
                        max_tokens=max(unit_tokens, self.config.max_tokens),
                        extra_flags=[
                            "hsc_structure_aware",
                            "protected_block_over_target",
                            "kept_protected_block_intact",
                        ],
                        metadata={
                            "target_tokens": self.config.target_tokens,
                            "max_tokens": self.config.max_tokens,
                            "protected_block_type": unit.type,
                        },
                    )
                )
                index += 1
                continue

            if not protected and unit_tokens > self.config.max_tokens:
                flush()
                for part in sentence_chunks(text, max_tokens=self.config.max_tokens):
                    chunks.append(
                        make_chunk(
                            doc=doc,
                            strategy=self.strategy,
                            index=index,
                            text=part,
                            blocks=[unit],
                            min_tokens=self.config.min_tokens,
                            max_tokens=self.config.max_tokens,
                            extra_flags=[
                                "hsc_structure_aware",
                                "split_long_text_by_sentence",
                            ],
                            metadata={
                                "target_tokens": self.config.target_tokens,
                                "source_block_token_estimate": unit_tokens,
                            },
                        )
                    )
                    index += 1
                continue

            if buffer:
                would_exceed = buffer_tokens + unit_tokens > self.config.max_tokens
                should_respect_section = (
                    not same_section
                    and buffer_tokens >= self.config.min_tokens
                )
                if would_exceed or should_respect_section:
                    flag = "section_boundary_respected" if should_respect_section else "max_length_boundary"
                    flush([flag])

            buffer.append(unit)
            buffer_texts.append(text)
            buffer_tokens += unit_tokens

        flush(["final_flush"])
        return chunks

    def _merge_short_chunks(self, chunks: list[RagChunk]) -> list[RagChunk]:
        if not chunks:
            return []
        merged: list[RagChunk] = []
        index = 0
        while index < len(chunks):
            current = chunks[index]
            if (
                current.token_count < self.config.min_tokens
                and index + 1 < len(chunks)
                and self._merge_allowed(current, chunks[index + 1])
            ):
                nxt = chunks[index + 1]
                current = self._merge_pair(current, nxt)
                index += 2
            else:
                index += 1
            merged.append(current)
        return merged

    def _merge_allowed(self, left: RagChunk, right: RagChunk) -> bool:
        if left.token_count + right.token_count > self.config.max_tokens:
            return False
        if not left.title_path or not right.title_path:
            return True
        return left.title_path[0] == right.title_path[0]

    def _merge_pair(self, left: RagChunk, right: RagChunk) -> RagChunk:
        flags = list(dict.fromkeys(left.quality_flags + right.quality_flags + ["merged_short_chunk"]))
        metadata = dict(left.metadata)
        metadata["merged_with"] = right.chunk_id
        # Rebuild through dict to keep Pydantic validation but avoid source block loss.
        data = left.model_dump(mode="json")
        data["text"] = left.text + "\n\n" + right.text
        data["token_count"] = estimate_tokens(data["text"])
        data["source_blocks"] = left.source_blocks + [
            block_id for block_id in right.source_blocks if block_id not in left.source_blocks
        ]
        data["quality_flags"] = flags
        data["tags"] = list(dict.fromkeys(left.tags + right.tags))[:10]
        data["entity_tags"] = list(dict.fromkeys(left.entity_tags + right.entity_tags))
        data["summary"] = left.summary
        data["metadata"] = metadata
        data["source_anchor"]["last_block_id"] = right.source_anchor.last_block_id
        data["source_anchor"]["block_count"] = len(data["source_blocks"])
        for section in right.source_anchor.sections:
            if section not in data["source_anchor"]["sections"]:
                data["source_anchor"]["sections"].append(section)
        for asset in right.source_anchor.assets:
            if asset not in data["source_anchor"]["assets"]:
                data["source_anchor"]["assets"].append(asset)
        return RagChunk.model_validate(data)

    def _renumber(self, doc: GovernedDocument, chunks: list[RagChunk]) -> list[RagChunk]:
        renumbered: list[RagChunk] = []
        for index, chunk in enumerate(chunks, 1):
            data = chunk.model_dump(mode="json")
            data["chunk_id"] = f"{doc.doc_id}_{self.strategy}_chunk_{index:05d}"
            renumbered.append(RagChunk.model_validate(data))
        return renumbered

    def _same_title_path(self, left: GovernedBlock, right: GovernedBlock) -> bool:
        return tuple(left.title_path) == tuple(right.title_path)
