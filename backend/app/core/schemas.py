# -*- coding: utf-8 -*-
"""Core data contracts for the HSC-RAG project.

The key boundary is intentional: HSC-RAG consumes governed, normalized content
and packages it into RAG-ready chunks. It does not perform upstream parsing,
cleaning, or terminology normalization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


NormalizationStatus = Literal[
    "provided_by_dataset",
    "provided_by_upstream",
    "simulated_governed",
]

BlockType = Literal[
    "title",
    "abstract",
    "heading",
    "paragraph",
    "list",
    "table",
    "figure",
    "code",
    "formula",
    "caption",
    "unknown",
]


class SourceAnchor(BaseModel):
    """A traceable pointer back to the source record or source block."""

    dataset: str
    split: str | None = None
    source_doc_id: str
    section_name: str | None = None
    paragraph_index: int | None = None
    asset_file: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class GovernedBlock(BaseModel):
    """A normalized content block after upstream governance."""

    block_id: str
    doc_id: str
    type: BlockType
    text: str
    order: int
    level: int = 0
    title_path: list[str] = Field(default_factory=list)
    source_anchor: SourceAnchor
    parent_heading_id: str | None = None
    entity_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GovernedQuery(BaseModel):
    """A question and its evidence mapping for retrieval evaluation."""

    query_id: str
    doc_id: str
    dataset: str
    split: str
    question: str
    answer: str
    answer_type: str = "unknown"
    is_unanswerable: bool = False
    gold_block_ids: list[str] = Field(default_factory=list)
    gold_evidence_texts: list[str] = Field(default_factory=list)
    evidence_match_score: float | None = None
    question_type: str = "evidence_qa"
    difficulty: str | None = None
    source_question_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GovernedDocument(BaseModel):
    """The primary input contract for HSC-RAG.

    `normalization_status` records why this document can be treated as content
    after terminology/format governance. Public datasets are marked as
    `provided_by_dataset`; manually built display examples should be marked as
    `simulated_governed`.
    """

    doc_id: str
    dataset: str
    split: str
    source_doc_id: str
    title: str
    abstract: str | None = None
    normalization_status: NormalizationStatus
    term_policy: str = "dataset_provided"
    governance_stage: str = "post_normalization_packaging"
    schema_version: str = "hsc-govdoc-v1"
    blocks: list[GovernedBlock] = Field(default_factory=list)
    queries: list[GovernedQuery] = Field(default_factory=list)
    source_ref: dict[str, Any] = Field(default_factory=dict)
    conversion_warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GoldEvidenceRecord(BaseModel):
    """A JSONL-friendly evidence mapping record."""

    query_id: str
    doc_id: str
    dataset: str
    split: str
    question: str
    answer: str
    gold_block_ids: list[str]
    gold_evidence_texts: list[str]
    evidence_matches: list[dict[str, Any]] = Field(default_factory=list)
    is_unanswerable: bool = False


ChunkStrategy = Literal[
    "fixed",
    "hsc_rag",
    "recursive",
    "semantic",
]


class ChunkSourceAnchor(BaseModel):
    """Aggregated source range for a chunk."""

    dataset: str
    split: str | None = None
    source_doc_id: str
    sections: list[str] = Field(default_factory=list)
    first_block_id: str | None = None
    last_block_id: str | None = None
    block_count: int = 0
    assets: list[str] = Field(default_factory=list)


class RagChunk(BaseModel):
    """RAG-ready chunk produced by a chunking strategy."""

    chunk_id: str
    doc_id: str
    dataset: str
    split: str
    strategy: ChunkStrategy
    text: str
    token_count: int
    title_path: list[str] = Field(default_factory=list)
    source_blocks: list[str] = Field(default_factory=list)
    source_anchor: ChunkSourceAnchor
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None
    entity_tags: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRunReport(BaseModel):
    """JSON-serializable report for a chunking run."""

    strategy: ChunkStrategy
    input_path: str
    output_path: str
    documents: int
    chunks: int
    total_tokens: int
    avg_tokens: float | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    quality_flag_counts: dict[str, int] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
