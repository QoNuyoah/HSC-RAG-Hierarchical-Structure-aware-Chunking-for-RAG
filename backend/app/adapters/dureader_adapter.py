# -*- coding: utf-8 -*-
"""DuReader -> GovernedDocument adapter.

DuReader is a Chinese machine-reading-comprehension dataset. In this project we
use the official preprocessed files as governed public data: each QA example is
converted into one `GovernedDocument` whose blocks are the retrieved document
paragraphs. The answer-related paragraph annotations (`is_selected` and
`most_related_para`) are converted into gold evidence block ids for chunk-level
retrieval evaluation.
"""

from __future__ import annotations

import csv
import hashlib
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


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def safe_id(value: str) -> str:
    value = NON_ID_RE.sub("_", value).strip("_")
    return value or hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]


def compact_answer(answers: list[Any], limit: int = 3) -> str:
    items = [normalize_text(answer) for answer in answers if normalize_text(answer)]
    return "；".join(items[:limit])


@dataclass
class DuReaderConversionStats:
    source: str
    split: str
    documents: int = 0
    blocks: int = 0
    queries: int = 0
    answerable_queries: int = 0
    unanswerable_queries: int = 0
    evidence_items: int = 0
    matched_evidence_items: int = 0
    selected_documents: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": f"dureader_{self.source}",
            "source": self.source,
            "split": self.split,
            "documents": self.documents,
            "blocks": self.blocks,
            "queries": self.queries,
            "answerable_queries": self.answerable_queries,
            "unanswerable_queries": self.unanswerable_queries,
            "evidence_items": self.evidence_items,
            "matched_evidence_items": self.matched_evidence_items,
            "evidence_match_rate": (
                self.matched_evidence_items / self.evidence_items
                if self.evidence_items
                else None
            ),
            "selected_documents": self.selected_documents,
            "warnings": self.warnings[:200],
        }


class DuReaderAdapter:
    """Convert DuReader preprocessed JSONL records to HSC-RAG contracts."""

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"DuReader zip not found: {self.zip_path}")

    def inner_path(self, *, source: str, split: str) -> str:
        if source not in {"search", "zhidao"}:
            raise ValueError("--source must be one of: search, zhidao")
        split_dir = {"train": "trainset", "dev": "devset", "test": "testset"}.get(split)
        if not split_dir:
            raise ValueError("--split must be one of: train, dev, test")
        return f"preprocessed/{split_dir}/{source}.{split}.json"

    def iter_records(
        self,
        *,
        source: str = "search",
        split: str = "dev",
        limit_docs: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        path = self.inner_path(source=source, split=split)
        with zipfile.ZipFile(self.zip_path) as zf:
            if path not in zf.namelist():
                raise FileNotFoundError(f"{path} not found in {self.zip_path}")
            with zf.open(path) as file:
                for index, raw in enumerate(file):
                    if limit_docs is not None and index >= limit_docs:
                        break
                    line = raw.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)

    def convert(
        self,
        *,
        source: str = "search",
        split: str = "dev",
        limit_docs: int | None = None,
    ) -> tuple[list[GovernedDocument], list[GoldEvidenceRecord], DuReaderConversionStats]:
        stats = DuReaderConversionStats(source=source, split=split)
        docs: list[GovernedDocument] = []
        evidence: list[GoldEvidenceRecord] = []

        for row_index, record in enumerate(self.iter_records(source=source, split=split, limit_docs=limit_docs)):
            try:
                doc, gold = self.convert_record(record, source=source, split=split)
            except Exception as exc:
                stats.warnings.append(f"row={row_index}: conversion failed: {type(exc).__name__}: {exc}")
                continue
            docs.append(doc)
            evidence.extend(gold)
            stats.documents += 1
            stats.blocks += len(doc.blocks)
            stats.queries += len(doc.queries)
            stats.answerable_queries += sum(1 for query in doc.queries if not query.is_unanswerable)
            stats.unanswerable_queries += sum(1 for query in doc.queries if query.is_unanswerable)
            stats.evidence_items += sum(len(item.gold_evidence_texts) for item in gold)
            stats.matched_evidence_items += sum(
                len([match for match in item.evidence_matches if match.get("block_id")])
                for item in gold
            )
            stats.selected_documents += sum(
                1 for document in record.get("documents", []) if document.get("is_selected")
            )
            stats.warnings.extend(doc.conversion_warnings)

        return docs, evidence, stats

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        source: str,
        split: str,
    ) -> tuple[GovernedDocument, list[GoldEvidenceRecord]]:
        raw_question_id = normalize_text(record.get("question_id"))
        safe_question_id = safe_id(raw_question_id)
        dataset = f"dureader_{source}"
        doc_id = f"{dataset}_{split}_{safe_question_id}"
        question = normalize_text(record.get("question"))
        warnings: list[str] = []

        blocks: list[GovernedBlock] = []
        block_by_doc_para: dict[tuple[int, int], GovernedBlock] = {}
        order = 1
        for doc_index, document in enumerate(record.get("documents", [])):
            title = normalize_text(document.get("title")) or f"document_{doc_index}"
            paragraphs = document.get("paragraphs") or []
            segmented_title = document.get("segmented_title") or []
            for para_index, paragraph in enumerate(paragraphs):
                text = normalize_text(paragraph)
                if not text:
                    continue
                block_id = f"{doc_id}_d{doc_index:02d}_p{para_index:03d}"
                block = GovernedBlock(
                    block_id=block_id,
                    doc_id=doc_id,
                    type="paragraph",
                    text=text,
                    order=order,
                    level=1,
                    title_path=[f"候选文档{doc_index + 1}: {title}"],
                    source_anchor=SourceAnchor(
                        dataset=dataset,
                        split=split,
                        source_doc_id=raw_question_id,
                        section_name=title,
                        paragraph_index=para_index,
                        extra={
                            "source": source,
                            "candidate_doc_index": doc_index,
                            "is_selected": bool(document.get("is_selected")),
                            "most_related_para": document.get("most_related_para"),
                        },
                    ),
                    entity_tags=[
                        normalize_text(token)
                        for token in segmented_title[:8]
                        if normalize_text(token) and normalize_text(token) not in {"_", "-", "|"}
                    ],
                    metadata={
                        "candidate_doc_index": doc_index,
                        "candidate_title": title,
                        "is_selected": bool(document.get("is_selected")),
                        "most_related_para": document.get("most_related_para"),
                    },
                )
                blocks.append(block)
                block_by_doc_para[(doc_index, para_index)] = block
                order += 1

        gold_blocks = self._gold_blocks(record, block_by_doc_para, warnings)
        answers = record.get("answers") or []
        answer_text = compact_answer(answers)
        is_unanswerable = not answer_text or not gold_blocks
        query_id = f"{doc_id}_q"
        query = GovernedQuery(
            query_id=query_id,
            doc_id=doc_id,
            dataset=dataset,
            split=split,
            question=question,
            answer=answer_text,
            answer_type=normalize_text(record.get("question_type")) or "unknown",
            is_unanswerable=is_unanswerable,
            gold_block_ids=[block.block_id for block in gold_blocks],
            gold_evidence_texts=[block.text for block in gold_blocks],
            evidence_match_score=max(record.get("match_scores") or [1.0]) if gold_blocks else None,
            question_type="dureader_mrc",
            source_question_id=raw_question_id,
            metadata={
                "source": source,
                "question_type": record.get("question_type"),
                "fact_or_opinion": record.get("fact_or_opinion"),
                "answer_docs": record.get("answer_docs") or [],
                "answer_spans": record.get("answer_spans") or [],
                "fake_answers": record.get("fake_answers") or [],
                "yesno_answers": record.get("yesno_answers") or [],
                "entity_answers": record.get("entity_answers") or [],
                "segmented_question": record.get("segmented_question") or [],
            },
        )

        doc = GovernedDocument(
            doc_id=doc_id,
            dataset=dataset,
            split=split,
            source_doc_id=raw_question_id,
            title=question,
            abstract=answer_text or None,
            normalization_status="provided_by_dataset",
            term_policy="dureader_preprocessed_segmentation",
            governance_stage="post_normalization_packaging",
            blocks=blocks,
            queries=[query],
            source_ref={
                "zip_path": self.zip_path.name,
                "inner_path": self.inner_path(source=source, split=split),
                "question_id": raw_question_id,
            },
            conversion_warnings=warnings,
            metadata={
                "source": source,
                "question_type": record.get("question_type"),
                "fact_or_opinion": record.get("fact_or_opinion"),
                "candidate_documents": len(record.get("documents", [])),
            },
        )
        gold_record = GoldEvidenceRecord(
            query_id=query.query_id,
            doc_id=doc.doc_id,
            dataset=dataset,
            split=split,
            question=query.question,
            answer=query.answer,
            gold_block_ids=query.gold_block_ids,
            gold_evidence_texts=query.gold_evidence_texts,
            evidence_matches=[
                {
                    "evidence_text": block.text,
                    "block_id": block.block_id,
                    "score": 1.0,
                    "method": "dureader_selected_most_related_para",
                }
                for block in gold_blocks
            ],
            is_unanswerable=query.is_unanswerable,
        )
        return doc, [gold_record]

    def _gold_blocks(
        self,
        record: dict[str, Any],
        block_by_doc_para: dict[tuple[int, int], GovernedBlock],
        warnings: list[str],
    ) -> list[GovernedBlock]:
        gold: list[GovernedBlock] = []
        documents = record.get("documents", [])

        def add(doc_index: int, para_index: int, reason: str) -> None:
            block = block_by_doc_para.get((doc_index, para_index))
            if block and block.block_id not in {item.block_id for item in gold}:
                gold.append(block)
            elif not block:
                warnings.append(f"missing gold block: doc={doc_index}, para={para_index}, reason={reason}")

        for doc_index, document in enumerate(documents):
            if not document.get("is_selected"):
                continue
            para_index = document.get("most_related_para")
            if isinstance(para_index, int):
                add(doc_index, para_index, "selected_most_related_para")

        if gold:
            return gold

        for doc_index in record.get("answer_docs") or []:
            if not isinstance(doc_index, int) or doc_index >= len(documents):
                continue
            para_index = documents[doc_index].get("most_related_para")
            if isinstance(para_index, int):
                add(doc_index, para_index, "answer_doc_most_related_para")

        return gold


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
