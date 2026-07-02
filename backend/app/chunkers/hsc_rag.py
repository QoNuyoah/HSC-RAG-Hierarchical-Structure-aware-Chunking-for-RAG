# -*- coding: utf-8 -*-
"""HSC-RAG: Hierarchical Structure-aware Chunking for RAG."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

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
    adaptive_boundary: bool = True
    semantic_boundary_scoring: bool = True
    semantic_boundary_threshold: float = 0.62
    semantic_soft_boundary_threshold: float = 0.52
    semantic_distance_threshold: float = 0.72
    semantic_window_blocks: int = 3
    structure_signal_weight: float = 0.45
    semantic_signal_weight: float = 0.35
    length_signal_weight: float = 0.20


@dataclass(frozen=True)
class BoundaryDecision:
    should_split: bool
    reason: str
    score: float
    signals: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "should_split": self.should_split,
            "split_reason": self.reason,
            "boundary_score": self.score,
            "signals": self.signals,
        }


TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]"
)
BOUNDARY_POLICY_VERSION = "hsc-rag-boundary-v2"


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
        base_config = self.config
        effective_config, adaptive_metadata = self._adaptive_config_for_document(units)
        self.config = effective_config
        try:
            initial = self._build_initial_chunks(doc, units, adaptive_metadata=adaptive_metadata)
            if self.config.merge_short_chunks:
                initial = self._merge_short_chunks(initial)
            return self._renumber(doc, initial)
        finally:
            self.config = base_config

    def _build_initial_chunks(
        self,
        doc: GovernedDocument,
        units: list[GovernedBlock],
        *,
        adaptive_metadata: dict[str, Any] | None = None,
    ) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        buffer: list[GovernedBlock] = []
        buffer_texts: list[str] = []
        buffer_tokens = 0
        index = 1

        def flush(
            extra_flags: list[str] | None = None,
            decision: BoundaryDecision | None = None,
        ) -> None:
            nonlocal buffer, buffer_texts, buffer_tokens, index
            if not buffer:
                buffer_texts = []
                buffer_tokens = 0
                return
            text = "\n\n".join(buffer_texts)
            metadata: dict[str, Any] = {
                "target_tokens": self.config.target_tokens,
                "max_tokens": self.config.max_tokens,
                "include_title_context": self.config.include_title_context,
                "boundary_policy": self._boundary_policy_metadata(),
            }
            if adaptive_metadata:
                metadata["adaptive_boundary"] = adaptive_metadata
            flags = list(extra_flags or [])
            if adaptive_metadata:
                flags.append("hsc_adaptive_boundary")
            if decision is not None:
                metadata["closing_boundary_decision"] = decision.as_metadata()
                flags.append("hsc_boundary_scored")
                if decision.signals.get("semantic_boundary_triggered"):
                    flags.append("semantic_boundary")
                elif decision.signals.get("semantic_distance", 0.0) >= self.config.semantic_distance_threshold:
                    flags.append("semantic_shift_observed")
            chunks.append(
                make_chunk(
                    doc=doc,
                    strategy=self.strategy,
                    index=index,
                    text=text,
                    blocks=buffer,
                    min_tokens=self.config.min_tokens,
                    max_tokens=self.config.max_tokens,
                    extra_flags=flags + ["hsc_structure_aware"],
                    metadata=metadata,
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
                        ]
                        + (["hsc_adaptive_boundary"] if adaptive_metadata else []),
                        metadata={
                            "target_tokens": self.config.target_tokens,
                            "max_tokens": self.config.max_tokens,
                            "boundary_policy": self._boundary_policy_metadata(),
                            "protected_block_type": unit.type,
                            **({"adaptive_boundary": adaptive_metadata} if adaptive_metadata else {}),
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
                            ]
                            + (["hsc_adaptive_boundary"] if adaptive_metadata else []),
                            metadata={
                                "target_tokens": self.config.target_tokens,
                                "source_block_token_estimate": unit_tokens,
                                "boundary_policy": self._boundary_policy_metadata(),
                                **({"adaptive_boundary": adaptive_metadata} if adaptive_metadata else {}),
                            },
                        )
                    )
                    index += 1
                continue

            if buffer:
                would_exceed = buffer_tokens + unit_tokens > self.config.max_tokens
                decision = self._score_boundary(
                    buffer=buffer,
                    next_block=unit,
                    buffer_tokens=buffer_tokens,
                    next_tokens=unit_tokens,
                    same_section=same_section,
                    would_exceed=would_exceed,
                    next_is_protected=protected,
                )
                if decision.should_split:
                    flush([decision.reason], decision=decision)

            buffer.append(unit)
            buffer_texts.append(text)
            buffer_tokens += unit_tokens

        flush(["final_flush"])
        return chunks

    def _adaptive_config_for_document(
        self,
        units: list[GovernedBlock],
    ) -> tuple[HscRagConfig, dict[str, Any] | None]:
        if not self.config.adaptive_boundary or not units:
            return self.config, None

        stats = self._document_boundary_stats(units)
        effective = self.config
        profile = "balanced"
        boundary_strength = "standard"
        reason = "default_thresholds"

        context_advantage = stats["context_need"] - stats["structure_need"]
        compact_blocks = stats["avg_block_tokens"] <= self.config.target_tokens * 0.45
        structure_dominant = (
            stats["structure_need"] >= 0.62
            or stats["top_title_transition_rate"] >= 0.20
            or (
                stats["title_transition_rate"] >= 0.42
                and stats["avg_adjacent_semantic_distance"] >= 0.58
            )
        )

        if structure_dominant and context_advantage < 0.24:
            profile = "structure_preserving"
            boundary_strength = "strong"
            reason = "structure_transitions_or_semantic_shifts_dominate"
            effective = replace(
                self.config,
                semantic_boundary_threshold=min(self.config.semantic_boundary_threshold, 0.60),
                semantic_soft_boundary_threshold=min(self.config.semantic_soft_boundary_threshold, 0.50),
                semantic_distance_threshold=min(self.config.semantic_distance_threshold, 0.70),
            )
        elif compact_blocks and context_advantage >= 0.34 and stats["structure_need"] < 0.54:
            profile = "high_context_coverage"
            boundary_strength = "loose"
            reason = "short_cohesive_blocks_need_more_context"
            effective = replace(
                self.config,
                target_tokens=max(
                    self.config.target_tokens,
                    min(self.config.max_tokens, int(self.config.target_tokens * 1.40)),
                ),
                semantic_boundary_threshold=max(self.config.semantic_boundary_threshold, 0.86),
                semantic_soft_boundary_threshold=max(self.config.semantic_soft_boundary_threshold, 0.76),
                semantic_distance_threshold=max(self.config.semantic_distance_threshold, 0.92),
            )
        elif context_advantage >= 0.18 and stats["structure_need"] < 0.62:
            profile = "context_coverage"
            boundary_strength = "relaxed"
            reason = "context_pressure_exceeds_structure_pressure"
            effective = replace(
                self.config,
                target_tokens=max(
                    self.config.target_tokens,
                    min(self.config.max_tokens, int(self.config.target_tokens * 1.25)),
                ),
                semantic_boundary_threshold=max(self.config.semantic_boundary_threshold, 0.76),
                semantic_soft_boundary_threshold=max(self.config.semantic_soft_boundary_threshold, 0.66),
                semantic_distance_threshold=max(self.config.semantic_distance_threshold, 0.84),
            )

        metadata = {
            "enabled": True,
            "profile": profile,
            "boundary_strength": boundary_strength,
            "method": "document_statistics",
            "decision_reason": reason,
            "stats": stats,
            "decision_basis": {
                "context_advantage": round(context_advantage, 4),
                "compact_blocks": compact_blocks,
                "structure_dominant": structure_dominant,
                "uses_dataset_name": False,
                "uses_gold_evidence": False,
                "uses_query_text": False,
            },
            "base_config": self._config_snapshot(self.config),
            "effective_config": self._config_snapshot(effective),
        }
        return effective, metadata

    def _document_boundary_stats(self, units: list[GovernedBlock]) -> dict[str, Any]:
        token_counts = [
            estimate_tokens(block_text(unit, include_title_path=self.config.include_title_context))
            for unit in units
        ]
        block_count = len(units)
        avg_tokens = round(sum(token_counts) / block_count, 4) if block_count else 0.0
        median_tokens = self._percentile(token_counts, 0.50)
        p75_tokens = self._percentile(token_counts, 0.75)
        short_block_rate = self._rate(token < self.config.min_tokens for token in token_counts)
        under_target_rate = self._rate(token < self.config.target_tokens * 0.50 for token in token_counts)
        protected_block_rate = self._rate(unit.type in PROTECTED_BLOCK_TYPES for unit in units)

        transitions = max(0, block_count - 1)
        title_changes = 0
        top_title_changes = 0
        semantic_distances: list[float] = []
        for left, right in zip(units, units[1:]):
            if tuple(left.title_path) != tuple(right.title_path):
                title_changes += 1
            if left.title_path and right.title_path and left.title_path[0] != right.title_path[0]:
                top_title_changes += 1
            semantic_distances.append(round(1.0 - self._semantic_similarity([left], right), 4))

        title_transition_rate = round(title_changes / transitions, 4) if transitions else 0.0
        top_title_transition_rate = round(top_title_changes / transitions, 4) if transitions else 0.0
        avg_semantic_distance = (
            round(sum(semantic_distances) / len(semantic_distances), 4)
            if semantic_distances
            else 0.0
        )
        p75_semantic_distance = self._percentile(semantic_distances, 0.75)
        semantic_continuity = round(max(0.0, 1.0 - avg_semantic_distance), 4)

        block_density = min(1.0, block_count / 80)
        context_need = round(
            min(
                1.0,
                0.30 * short_block_rate
                + 0.22 * under_target_rate
                + 0.20 * block_density
                + 0.18 * semantic_continuity
                + 0.10 * protected_block_rate,
            ),
            4,
        )
        structure_need = round(
            min(
                1.0,
                0.36 * top_title_transition_rate
                + 0.24 * title_transition_rate
                + 0.24 * avg_semantic_distance
                + 0.10 * p75_semantic_distance
                + 0.06 * protected_block_rate,
            ),
            4,
        )

        return {
            "block_count": block_count,
            "avg_block_tokens": avg_tokens,
            "median_block_tokens": median_tokens,
            "p75_block_tokens": p75_tokens,
            "short_block_rate": short_block_rate,
            "under_half_target_rate": under_target_rate,
            "protected_block_rate": protected_block_rate,
            "title_transition_rate": title_transition_rate,
            "top_title_transition_rate": top_title_transition_rate,
            "avg_adjacent_semantic_distance": avg_semantic_distance,
            "p75_adjacent_semantic_distance": p75_semantic_distance,
            "semantic_continuity": semantic_continuity,
            "context_need": context_need,
            "structure_need": structure_need,
        }

    def _percentile(self, values: list[int | float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
        return round(float(ordered[index]), 4)

    def _rate(self, items: Iterable[bool]) -> float:
        values = list(items)
        return round(sum(1 for item in values if item) / len(values), 4) if values else 0.0

    def _config_snapshot(self, config: HscRagConfig) -> dict[str, Any]:
        return {
            "min_tokens": config.min_tokens,
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "semantic_boundary_threshold": config.semantic_boundary_threshold,
            "semantic_soft_boundary_threshold": config.semantic_soft_boundary_threshold,
            "semantic_distance_threshold": config.semantic_distance_threshold,
            "semantic_window_blocks": config.semantic_window_blocks,
            "structure_signal_weight": config.structure_signal_weight,
            "semantic_signal_weight": config.semantic_signal_weight,
            "length_signal_weight": config.length_signal_weight,
        }

    def _score_boundary(
        self,
        *,
        buffer: list[GovernedBlock],
        next_block: GovernedBlock,
        buffer_tokens: int,
        next_tokens: int,
        same_section: bool,
        would_exceed: bool,
        next_is_protected: bool,
    ) -> BoundaryDecision:
        left = buffer[-1]
        title_path_changed = tuple(left.title_path) != tuple(next_block.title_path)
        top_title_changed = bool(
            left.title_path
            and next_block.title_path
            and left.title_path[0] != next_block.title_path[0]
        )
        block_type_changed = left.type != next_block.type
        semantic_similarity = self._semantic_similarity(buffer, next_block)
        semantic_distance = round(1.0 - semantic_similarity, 4)
        structure_signal = self._structure_signal(
            title_path_changed=title_path_changed,
            top_title_changed=top_title_changed,
            block_type_changed=block_type_changed,
            same_section=same_section,
            next_is_protected=next_is_protected,
        )
        length_pressure = round(
            min(1.0, buffer_tokens / max(1, self.config.target_tokens)),
            4,
        )
        score = round(
            (
                self.config.structure_signal_weight * structure_signal
                + self.config.semantic_signal_weight * semantic_distance
                + self.config.length_signal_weight * length_pressure
            ),
            4,
        )

        semantic_boundary_triggered = (
            self.config.semantic_boundary_scoring
            and semantic_distance >= self.config.semantic_distance_threshold
            and buffer_tokens >= self.config.min_tokens
            and length_pressure >= 0.65
        )
        section_boundary_triggered = (
            not same_section
            and buffer_tokens >= self.config.min_tokens
            and score >= self.config.semantic_soft_boundary_threshold
        )
        scored_boundary_triggered = (
            self.config.semantic_boundary_scoring
            and buffer_tokens >= self.config.min_tokens
            and score >= self.config.semantic_boundary_threshold
        )
        target_length_boundary = (
            buffer_tokens >= self.config.target_tokens
            and score >= self.config.semantic_soft_boundary_threshold
        )

        if would_exceed:
            should_split = True
            reason = "max_length_boundary"
        elif semantic_boundary_triggered:
            should_split = True
            reason = "semantic_boundary"
        elif section_boundary_triggered:
            should_split = True
            reason = "section_boundary_respected"
        elif scored_boundary_triggered:
            should_split = True
            reason = "scored_structure_semantic_boundary"
        elif target_length_boundary:
            should_split = True
            reason = "target_length_boundary"
        else:
            should_split = False
            reason = "continue_accumulating"

        signals = {
            "policy_version": BOUNDARY_POLICY_VERSION,
            "structure_signal": structure_signal,
            "semantic_similarity": semantic_similarity,
            "semantic_distance": semantic_distance,
            "length_pressure": length_pressure,
            "buffer_tokens": buffer_tokens,
            "next_block_tokens": next_tokens,
            "target_tokens": self.config.target_tokens,
            "max_tokens": self.config.max_tokens,
            "same_section": same_section,
            "title_path_changed": title_path_changed,
            "top_title_changed": top_title_changed,
            "block_type_changed": block_type_changed,
            "next_block_type": next_block.type,
            "next_is_protected": next_is_protected,
            "would_exceed_max_tokens": would_exceed,
            "semantic_boundary_triggered": semantic_boundary_triggered,
            "section_boundary_triggered": section_boundary_triggered,
            "scored_boundary_triggered": scored_boundary_triggered,
            "target_length_boundary": target_length_boundary,
            "weights": {
                "structure": self.config.structure_signal_weight,
                "semantic": self.config.semantic_signal_weight,
                "length": self.config.length_signal_weight,
            },
            "thresholds": {
                "boundary_score": self.config.semantic_boundary_threshold,
                "soft_boundary_score": self.config.semantic_soft_boundary_threshold,
                "semantic_distance": self.config.semantic_distance_threshold,
            },
            "left_block_id": left.block_id,
            "right_block_id": next_block.block_id,
            "left_title_path": left.title_path,
            "right_title_path": next_block.title_path,
        }
        return BoundaryDecision(
            should_split=should_split,
            reason=reason,
            score=score,
            signals=signals,
        )

    def _structure_signal(
        self,
        *,
        title_path_changed: bool,
        top_title_changed: bool,
        block_type_changed: bool,
        same_section: bool,
        next_is_protected: bool,
    ) -> float:
        if top_title_changed:
            return 1.0
        if title_path_changed:
            return 0.85
        if not same_section:
            return 0.7
        if next_is_protected:
            return 0.45
        if block_type_changed:
            return 0.25
        return 0.0

    def _semantic_similarity(
        self,
        buffer: list[GovernedBlock],
        next_block: GovernedBlock,
    ) -> float:
        if not self.config.semantic_boundary_scoring:
            return 1.0
        left_blocks = buffer[-max(1, self.config.semantic_window_blocks) :]
        left_text = " ".join(block_text(block, include_title_path=True) for block in left_blocks)
        right_text = block_text(next_block, include_title_path=True)
        left_counter = self._token_counter(left_text)
        right_counter = self._token_counter(right_text)
        if not left_counter or not right_counter:
            return 0.0
        dot = sum(left_counter[token] * right_counter.get(token, 0) for token in left_counter)
        left_norm = math.sqrt(sum(value * value for value in left_counter.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counter.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return round(max(0.0, min(1.0, dot / (left_norm * right_norm))), 4)

    def _token_counter(self, text: str) -> Counter[str]:
        tokens = [token.lower() for token in TOKEN_RE.findall(normalize_text(text))]
        return Counter(token for token in tokens if len(token) > 1 or not token.isascii())

    def _boundary_policy_metadata(self) -> dict[str, Any]:
        return {
            "version": BOUNDARY_POLICY_VERSION,
            "scoring_enabled": self.config.semantic_boundary_scoring,
            "semantic_method": "local_bow_cosine_over_recent_blocks",
            "semantic_window_blocks": self.config.semantic_window_blocks,
            "formula": "score = structure*w_s + semantic_distance*w_sem + length_pressure*w_l",
            "weights": {
                "structure": self.config.structure_signal_weight,
                "semantic": self.config.semantic_signal_weight,
                "length": self.config.length_signal_weight,
            },
            "thresholds": {
                "boundary_score": self.config.semantic_boundary_threshold,
                "soft_boundary_score": self.config.semantic_soft_boundary_threshold,
                "semantic_distance": self.config.semantic_distance_threshold,
            },
        }

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
