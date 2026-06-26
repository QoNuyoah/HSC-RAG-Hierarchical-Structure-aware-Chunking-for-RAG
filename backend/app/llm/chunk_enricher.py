# -*- coding: utf-8 -*-
"""LLM-assisted enrichment for HSC-RAG chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.chunkers.common import normalize_text, summarize_text
from app.core.schemas import RagChunk
from app.llm.providers import JsonDict, JsonLlmProvider, LlmMessage


PROMPT_VERSION = "hsc-rag-enrich-v1"


@dataclass(frozen=True)
class ChunkEnrichmentConfig:
    prompt_version: str = PROMPT_VERSION
    max_input_chars: int = 4200
    max_topic_tags: int = 6
    max_entity_tags: int = 10
    include_qa: bool = False


class ChunkSemanticEnricher:
    """Add LLM semantic organization metadata after deterministic chunking."""

    def __init__(
        self,
        *,
        provider: JsonLlmProvider,
        config: ChunkEnrichmentConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or ChunkEnrichmentConfig()

    def enrich_chunk(self, chunk: RagChunk) -> RagChunk:
        fallback = self._fallback_enrichment(chunk)
        messages = self._build_messages(chunk)
        raw = self.provider.complete_json(messages, fallback=fallback)
        enrichment = self._normalize_enrichment(raw, fallback=fallback, chunk=chunk)

        metadata = dict(chunk.metadata or {})
        metadata["llm_enrichment"] = enrichment
        return chunk.model_copy(update={"metadata": metadata})

    def _build_messages(self, chunk: RagChunk) -> list[LlmMessage]:
        system = (
            "You are the semantic organization skill in HSC-RAG Agent. "
            "The chunk boundary has already been produced by a deterministic, "
            "auditable structure-aware chunker. Your job is only to organize "
            "the chunk for downstream RAG consumption: faithful summary, topic "
            "tags, entity tags, semantic integrity score, and quality reasons. "
            "Use only evidence in the given chunk. Do not invent facts. "
            "Return one strict JSON object."
        )
        schema_hint = {
            "summary": "faithful one-sentence or two-sentence summary",
            "topic_tags": ["3-6 concise topic tags"],
            "entity_tags": ["named methods, datasets, metrics, systems, or concepts"],
            "semantic_integrity_score": "0-5 float",
            "summary_faithfulness_score": "0-5 float",
            "tag_accuracy_score": "0-5 float",
            "faithfulness_risk": "low|medium|high",
            "quality_reason": "short reason grounded in chunk text",
            "qa_pairs": [
                {
                    "instruction": "answer using only the chunk",
                    "question": "optional QA question",
                    "answer": "faithful answer",
                    "answerability": "answerable|unanswerable",
                    "faithfulness_score": "0-5 float",
                }
            ],
        }
        qa_instruction = (
            "Also generate exactly one answerable QA/instruction sample."
            if self.config.include_qa
            else "Set qa_pairs to an empty list."
        )
        user = {
            "expected_json_schema": schema_hint,
            "task_instruction": qa_instruction,
            "chunk": {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "dataset": chunk.dataset,
                "split": chunk.split,
                "title_path": chunk.title_path,
                "token_count": chunk.token_count,
                "source_blocks": chunk.source_blocks,
                "source_anchor": chunk.source_anchor.model_dump(mode="json"),
                "existing_rule_tags": chunk.tags,
                "existing_rule_summary": chunk.summary,
                "existing_rule_entity_tags": chunk.entity_tags,
                "quality_flags": chunk.quality_flags,
                "text": _truncate(chunk.text, self.config.max_input_chars),
            },
        }
        return [
            LlmMessage(role="system", content=system),
            LlmMessage(role="user", content=_json_dumps(user)),
        ]

    def _fallback_enrichment(self, chunk: RagChunk) -> JsonDict:
        summary = normalize_text(chunk.summary or summarize_text(chunk.text, max_words=60))
        topic_tags = _dedupe([*chunk.title_path, *chunk.tags], limit=self.config.max_topic_tags)
        entity_tags = _dedupe(chunk.entity_tags, limit=self.config.max_entity_tags)
        integrity = _heuristic_integrity_score(chunk)
        tag_score = 4.5 if topic_tags else 3.8
        faithfulness = 4.7 if summary else 4.0
        risk = "low"
        if "long_chunk" in chunk.quality_flags or "mixed_title_paths" in chunk.quality_flags:
            risk = "medium"
        if "source_anchor_complete" not in chunk.quality_flags:
            risk = "high"

        qa_pairs = []
        if self.config.include_qa:
            qa_topic = _qa_topic(topic_tags, chunk)
            qa_pairs.append(
                {
                    "instruction": "Answer the question using only the provided chunk.",
                    "question": f"According to the chunk, what is the main point about {qa_topic}?",
                    "answer": summary,
                    "answerability": "answerable" if summary else "unanswerable",
                    "faithfulness_score": faithfulness,
                    "evidence_chunk_id": chunk.chunk_id,
                    "evidence_source_blocks": chunk.source_blocks[:5],
                }
            )

        return {
            "schema_version": "hsc-rag-llm-enrichment-v1",
            "skill": "llm_semantic_organization",
            "prompt_version": self.config.prompt_version,
            "provider": self.provider.provider_name,
            "model": self.provider.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "topic_tags": topic_tags,
            "entity_tags": entity_tags,
            "semantic_integrity_score": integrity,
            "summary_faithfulness_score": faithfulness,
            "tag_accuracy_score": tag_score,
            "faithfulness_risk": risk,
            "quality_reason": _quality_reason(chunk, integrity, risk),
            "source_anchor_complete": "source_anchor_complete" in chunk.quality_flags,
            "qa_pairs": qa_pairs,
        }

    def _normalize_enrichment(self, raw: JsonDict, *, fallback: JsonDict, chunk: RagChunk) -> JsonDict:
        result = dict(fallback)
        result.update({key: value for key, value in raw.items() if value is not None})
        result["schema_version"] = "hsc-rag-llm-enrichment-v1"
        result["skill"] = "llm_semantic_organization"
        result["prompt_version"] = self.config.prompt_version
        result["provider"] = self.provider.provider_name
        result["model"] = self.provider.model
        result["generated_at"] = result.get("generated_at") or datetime.now(timezone.utc).isoformat()
        result["summary"] = normalize_text(str(result.get("summary") or fallback["summary"]))
        result["topic_tags"] = _dedupe(_as_list(result.get("topic_tags")), limit=self.config.max_topic_tags)
        result["entity_tags"] = _dedupe(_as_list(result.get("entity_tags")), limit=self.config.max_entity_tags)
        result["semantic_integrity_score"] = _score(result.get("semantic_integrity_score"), fallback["semantic_integrity_score"])
        result["summary_faithfulness_score"] = _score(
            result.get("summary_faithfulness_score"),
            fallback["summary_faithfulness_score"],
        )
        result["tag_accuracy_score"] = _score(result.get("tag_accuracy_score"), fallback["tag_accuracy_score"])
        if result.get("faithfulness_risk") not in {"low", "medium", "high"}:
            result["faithfulness_risk"] = fallback["faithfulness_risk"]
        result["quality_reason"] = normalize_text(str(result.get("quality_reason") or fallback["quality_reason"]))
        result["source_anchor_complete"] = "source_anchor_complete" in chunk.quality_flags
        result["qa_pairs"] = _normalize_qa_pairs(result.get("qa_pairs"), chunk)
        return result


def _heuristic_integrity_score(chunk: RagChunk) -> float:
    score = 4.6
    if "title_path_consistent" in chunk.quality_flags:
        score += 0.2
    if "section_boundary_respected" in chunk.quality_flags:
        score += 0.1
    if "mixed_title_paths" in chunk.quality_flags:
        score -= 0.4
    if "short_chunk" in chunk.quality_flags:
        score -= 0.3
    if "long_chunk" in chunk.quality_flags:
        score -= 0.5
    if not chunk.text.strip():
        score = 0.0
    return round(max(0.0, min(5.0, score)), 2)


def _quality_reason(chunk: RagChunk, score: float, risk: str) -> str:
    sections = " > ".join(chunk.title_path) if chunk.title_path else "untitled section"
    return (
        f"Chunk keeps source anchors and is organized around {sections}; "
        f"quality flags={','.join(chunk.quality_flags)}; "
        f"semantic_integrity_score={score}; faithfulness_risk={risk}."
    )


def _qa_topic(topic_tags: list[str], chunk: RagChunk) -> str:
    weak_tags = {
        "abstract",
        "introduction",
        "related work",
        "paragraph",
        "list",
        "table",
        "figure",
        "caption",
        "formula",
        "code",
    }
    for tag in topic_tags:
        if tag.lower() not in weak_tags:
            return tag
    if chunk.title_path:
        return chunk.title_path[-1]
    return "the chunk content"


def _normalize_qa_pairs(value: Any, chunk: RagChunk) -> list[JsonDict]:
    pairs = value if isinstance(value, list) else []
    normalized = []
    for item in pairs[:3]:
        if not isinstance(item, dict):
            continue
        answerability = item.get("answerability")
        if answerability not in {"answerable", "unanswerable"}:
            answerability = "answerable" if item.get("answer") else "unanswerable"
        normalized.append(
            {
                "instruction": normalize_text(str(item.get("instruction") or "Answer using only the provided chunk.")),
                "question": normalize_text(str(item.get("question") or "")),
                "answer": normalize_text(str(item.get("answer") or "")),
                "answerability": answerability,
                "faithfulness_score": _score(item.get("faithfulness_score"), 4.5),
                "evidence_chunk_id": chunk.chunk_id,
                "evidence_source_blocks": chunk.source_blocks[:5],
            }
        )
    return normalized


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_text(str(item)) for item in value if normalize_text(str(item))]
    if isinstance(value, str) and value.strip():
        return [normalize_text(part) for part in value.split(",") if normalize_text(part)]
    return []


def _dedupe(items: list[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = normalize_text(str(item))
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _score(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float(default)
    return round(max(0.0, min(5.0, score)), 2)


def _truncate(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _json_dumps(data: JsonDict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)
