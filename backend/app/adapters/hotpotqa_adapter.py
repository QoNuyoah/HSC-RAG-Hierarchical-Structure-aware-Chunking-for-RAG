# -*- coding: utf-8 -*-
"""HotpotQA -> GovernedDocument adapter.

HotpotQA is useful as an advanced multi-hop QA supplement. Each QA example is
converted into one GovernedDocument. Context sentences become traceable blocks,
and supporting_facts are mapped to gold evidence block ids for retrieval eval.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.schemas import (
    GoldEvidenceRecord,
    GovernedBlock,
    GovernedDocument,
    GovernedQuery,
    SourceAnchor,
)


WHITESPACE_RE = re.compile(r"\s+")
NON_ID_RE = re.compile(r"[^0-9A-Za-z_.-]+")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_+\-]{2,}(?:\s+[A-Z][A-Za-z0-9_+\-]{2,}){0,3}\b")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def safe_id(value: str) -> str:
    value = NON_ID_RE.sub("_", value).strip("_")
    return value or hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]


def infer_member(zip_path: Path, member: str | None = None) -> str:
    if member:
        return member
    with zipfile.ZipFile(zip_path) as zf:
        json_members = [name for name in zf.namelist() if name.lower().endswith(".json")]
    if not json_members:
        raise FileNotFoundError(f"No JSON member found in {zip_path}")
    return json_members[0]


@dataclass
class HotpotQAConversionStats:
    split: str
    documents: int = 0
    blocks: int = 0
    queries: int = 0
    answerable_queries: int = 0
    evidence_items: int = 0
    matched_evidence_items: int = 0
    context_articles: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": "hotpotqa",
            "split": self.split,
            "documents": self.documents,
            "blocks": self.blocks,
            "queries": self.queries,
            "answerable_queries": self.answerable_queries,
            "evidence_items": self.evidence_items,
            "matched_evidence_items": self.matched_evidence_items,
            "evidence_match_rate": (
                self.matched_evidence_items / self.evidence_items
                if self.evidence_items
                else None
            ),
            "context_articles": self.context_articles,
            "warnings": self.warnings[:200],
        }


class HotpotQAAdapter:
    """Convert local HotpotQA zip artifacts into HSC-RAG contracts."""

    def __init__(self, zip_path: str | Path, member: str | None = None):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"HotpotQA zip not found: {self.zip_path}")
        self.member = infer_member(self.zip_path, member)

    def iter_records(self, *, limit_docs: int | None = None) -> Iterable[dict[str, Any]]:
        yielded = 0
        decoder = json.JSONDecoder()
        buffer = ""
        started = False

        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open(self.member) as raw:
                stream = io.TextIOWrapper(raw, encoding="utf-8")
                while True:
                    chunk = stream.read(65536)
                    if chunk:
                        buffer += chunk
                    elif not buffer.strip():
                        break

                    while True:
                        if not started:
                            buffer = buffer.lstrip()
                            if buffer.startswith("["):
                                buffer = buffer[1:]
                                started = True
                            else:
                                break
                        buffer = buffer.lstrip()
                        if buffer.startswith("]"):
                            return
                        if buffer.startswith(","):
                            buffer = buffer[1:].lstrip()
                        if not buffer:
                            break
                        try:
                            record, index = decoder.raw_decode(buffer)
                        except json.JSONDecodeError:
                            break
                        yield record
                        yielded += 1
                        if limit_docs is not None and yielded >= limit_docs:
                            return
                        buffer = buffer[index:]

                    if not chunk:
                        break

    def convert(
        self,
        *,
        split: str = "train",
        limit_docs: int | None = 50,
    ) -> tuple[list[GovernedDocument], list[GoldEvidenceRecord], HotpotQAConversionStats]:
        stats = HotpotQAConversionStats(split=split)
        docs: list[GovernedDocument] = []
        evidence: list[GoldEvidenceRecord] = []

        for row_index, record in enumerate(self.iter_records(limit_docs=limit_docs)):
            try:
                doc, gold = self.convert_record(record, split=split)
            except Exception as exc:
                stats.warnings.append(f"row={row_index}: conversion failed: {type(exc).__name__}: {exc}")
                continue
            docs.append(doc)
            evidence.extend(gold)
            stats.documents += 1
            stats.blocks += len(doc.blocks)
            stats.queries += len(doc.queries)
            stats.answerable_queries += sum(1 for query in doc.queries if not query.is_unanswerable)
            stats.evidence_items += sum(len(item.gold_evidence_texts) for item in gold)
            stats.matched_evidence_items += sum(
                len([match for match in item.evidence_matches if match.get("block_id")])
                for item in gold
            )
            stats.context_articles += sum(
                int(block.metadata.get("is_first_sentence", False))
                for block in doc.blocks
            )
            stats.warnings.extend(doc.conversion_warnings)

        return docs, evidence, stats

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        split: str,
    ) -> tuple[GovernedDocument, list[GoldEvidenceRecord]]:
        raw_id = normalize_text(record.get("_id"))
        doc_id = f"hotpotqa_{split}_{safe_id(raw_id)}"
        question = normalize_text(record.get("question"))
        answer = normalize_text(record.get("answer"))
        question_type = normalize_text(record.get("type")) or "unknown"
        level = normalize_text(record.get("level")) or None
        warnings: list[str] = []

        supporting_facts = [
            (normalize_text(item[0]), int(item[1]))
            for item in record.get("supporting_facts", [])
            if isinstance(item, list) and len(item) >= 2 and isinstance(item[1], int)
        ]
        supporting_set = set(supporting_facts)

        blocks: list[GovernedBlock] = []
        block_by_title_sentence: dict[tuple[str, int], GovernedBlock] = {}
        order = 1
        for context_index, item in enumerate(record.get("context", [])):
            if not isinstance(item, list) or len(item) < 2:
                warnings.append(f"context item {context_index} is malformed")
                continue
            title = normalize_text(item[0]) or f"context_{context_index}"
            sentences = item[1] if isinstance(item[1], list) else []
            for sentence_index, sentence in enumerate(sentences):
                text = normalize_text(sentence)
                if not text:
                    continue
                is_supporting = (title, sentence_index) in supporting_set
                block_id = f"{doc_id}_c{context_index:02d}_s{sentence_index:03d}"
                block = GovernedBlock(
                    block_id=block_id,
                    doc_id=doc_id,
                    type="paragraph",
                    text=text,
                    order=order,
                    level=1,
                    title_path=[title],
                    source_anchor=SourceAnchor(
                        dataset="hotpotqa",
                        split=split,
                        source_doc_id=raw_id,
                        section_name=title,
                        paragraph_index=sentence_index,
                        extra={
                            "context_index": context_index,
                            "sentence_index": sentence_index,
                            "is_supporting_fact": is_supporting,
                        },
                    ),
                    entity_tags=derive_entities(title, text),
                    metadata={
                        "context_index": context_index,
                        "sentence_index": sentence_index,
                        "context_title": title,
                        "is_first_sentence": sentence_index == 0,
                        "is_supporting_fact": is_supporting,
                    },
                )
                blocks.append(block)
                block_by_title_sentence[(title, sentence_index)] = block
                order += 1

        gold_blocks: list[GovernedBlock] = []
        for title, sentence_index in supporting_facts:
            block = block_by_title_sentence.get((title, sentence_index))
            if block and block.block_id not in {item.block_id for item in gold_blocks}:
                gold_blocks.append(block)
            else:
                warnings.append(f"missing supporting fact block: title={title}, sentence={sentence_index}")

        query_id = f"{doc_id}_q"
        query = GovernedQuery(
            query_id=query_id,
            doc_id=doc_id,
            dataset="hotpotqa",
            split=split,
            question=question,
            answer=answer,
            answer_type=question_type,
            is_unanswerable=not answer or not gold_blocks,
            gold_block_ids=[block.block_id for block in gold_blocks],
            gold_evidence_texts=[block.text for block in gold_blocks],
            evidence_match_score=(
                len(gold_blocks) / len(supporting_facts)
                if supporting_facts
                else None
            ),
            question_type="hotpotqa_multihop",
            difficulty=level,
            source_question_id=raw_id,
            metadata={
                "hotpotqa_type": question_type,
                "level": level,
                "supporting_facts": supporting_facts,
            },
        )

        doc = GovernedDocument(
            doc_id=doc_id,
            dataset="hotpotqa",
            split=split,
            source_doc_id=raw_id,
            title=question,
            normalization_status="provided_by_dataset",
            term_policy="dataset_provided",
            blocks=blocks,
            queries=[query],
            source_ref={
                "zip_path": str(self.zip_path),
                "member": self.member,
                "source_record_id": raw_id,
            },
            conversion_warnings=warnings,
            metadata={
                "adapter": "hotpotqa_adapter",
                "hotpotqa_type": question_type,
                "level": level,
                "context_articles": len(record.get("context", [])),
                "supporting_fact_count": len(supporting_facts),
            },
        )
        gold_record = GoldEvidenceRecord(
            query_id=query.query_id,
            doc_id=query.doc_id,
            dataset=query.dataset,
            split=query.split,
            question=query.question,
            answer=query.answer,
            gold_block_ids=query.gold_block_ids,
            gold_evidence_texts=query.gold_evidence_texts,
            evidence_matches=[
                {
                    "evidence_text": block.text,
                    "block_id": block.block_id,
                    "score": 1.0,
                    "method": "hotpotqa_supporting_facts",
                    "title": block.source_anchor.section_name,
                    "sentence_index": block.source_anchor.paragraph_index,
                }
                for block in gold_blocks
            ],
            is_unanswerable=query.is_unanswerable,
        )
        return doc, [gold_record]


def derive_entities(title: str, text: str, limit: int = 12) -> list[str]:
    entities: list[str] = []
    for value in [title]:
        value = normalize_text(value)
        if value and value not in entities:
            entities.append(value)
    for match in ENTITY_RE.finditer(text):
        value = normalize_text(match.group(0))
        if value and value not in entities:
            entities.append(value)
        if len(entities) >= limit:
            break
    return entities[:limit]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_queries_csv(path: str | Path, queries: Iterable[GovernedQuery]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "doc_id",
        "dataset",
        "split",
        "question",
        "answer",
        "answer_type",
        "is_unanswerable",
        "gold_block_ids",
        "gold_evidence_texts",
        "evidence_match_score",
        "question_type",
        "difficulty",
        "source_question_id",
        "metadata",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query in queries:
            row = query.model_dump()
            row["gold_block_ids"] = json.dumps(row["gold_block_ids"], ensure_ascii=False)
            row["gold_evidence_texts"] = json.dumps(row["gold_evidence_texts"], ensure_ascii=False)
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            writer.writerow(row)
