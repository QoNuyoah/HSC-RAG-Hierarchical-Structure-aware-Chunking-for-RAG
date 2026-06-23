# -*- coding: utf-8 -*-
"""QASPER -> GovernedDocument adapter.

This adapter treats QASPER as a public, already-curated dataset. It does not
claim to perform terminology normalization; instead it records
`normalization_status="provided_by_dataset"` so HSC-RAG can stay aligned with
the course requirement: chunking happens after upstream governance and during
result packaging.
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

import pandas as pd

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


def normalize_for_match(text: Any) -> str:
    return normalize_text(text).lower()


def safe_id(value: str) -> str:
    return NON_ID_RE.sub("_", value).strip("_")


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def token_estimate(text: str) -> int:
    """A lightweight token estimate used before the real tokenizer exists."""

    normalized = normalize_text(text)
    if not normalized:
        return 0
    # English QASPER text is whitespace-tokenized well enough for data prep.
    return len(normalized.split())


def _array(values: Iterable[Any], dtype: Any = None) -> list[Any]:
    return list(values)


SAFE_EVAL_GLOBALS = {"__builtins__": {}}
SAFE_EVAL_LOCALS = {
    "array": _array,
    "nan": None,
    "NaN": None,
    "None": None,
    "True": True,
    "False": False,
    # QASPER CSV cells contain numpy-like `dtype=object`; the adapter ignores it.
    "object": "object",
}


def parse_hf_csv_cell(value: Any) -> Any:
    """Parse HuggingFace CSV cells containing Python/numpy repr strings.

    The local QASPER.zip stores nested fields like:
    {'section_name': array([...], dtype=object), 'paragraphs': array([...])}

    This function accepts only a minimal evaluation environment and maps array()
    calls to Python lists. It is intended for curated local dataset files, not
    arbitrary untrusted user input.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return eval(text, SAFE_EVAL_GLOBALS, SAFE_EVAL_LOCALS)
    except Exception as exc:  # pragma: no cover - kept explicit for reports
        raise ValueError(f"Failed to parse QASPER CSV cell: {text[:200]}") from exc


@dataclass
class EvidenceMatch:
    evidence_text: str
    block_id: str | None
    score: float
    method: str


@dataclass
class ConversionStats:
    split: str
    documents: int = 0
    blocks: int = 0
    queries: int = 0
    answerable_queries: int = 0
    unanswerable_queries: int = 0
    evidence_items: int = 0
    matched_evidence_items: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "warnings": self.warnings[:200],
        }


class QasperAdapter:
    """Convert QASPER CSV-in-zip records to HSC-RAG GovernedDocument."""

    dataset_name = "qasper"

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"QASPER zip not found: {self.zip_path}")

    def available_splits(self) -> list[str]:
        with zipfile.ZipFile(self.zip_path) as zf:
            return sorted(Path(name).stem for name in zf.namelist() if name.endswith(".csv"))

    def load_split_frame(self, split: str, limit_docs: int | None = None) -> pd.DataFrame:
        csv_name = f"{split}.csv"
        with zipfile.ZipFile(self.zip_path) as zf:
            if csv_name not in zf.namelist():
                raise ValueError(
                    f"Split {split!r} not found in {self.zip_path}. "
                    f"Available: {self.available_splits()}"
                )
            data = zf.read(csv_name)
        return pd.read_csv(io.BytesIO(data), nrows=limit_docs)

    def iter_documents(
        self,
        split: str = "train",
        limit_docs: int | None = None,
    ) -> tuple[list[GovernedDocument], list[GoldEvidenceRecord], ConversionStats]:
        df = self.load_split_frame(split=split, limit_docs=limit_docs)
        stats = ConversionStats(split=split)
        documents: list[GovernedDocument] = []
        evidence_records: list[GoldEvidenceRecord] = []

        for row_index, row in df.iterrows():
            try:
                doc, doc_evidence = self.convert_row(row=row, split=split)
            except Exception as exc:
                stats.warnings.append(f"row={row_index}: conversion failed: {exc}")
                continue

            documents.append(doc)
            evidence_records.extend(doc_evidence)
            stats.documents += 1
            stats.blocks += len(doc.blocks)
            stats.queries += len(doc.queries)
            for query in doc.queries:
                if query.is_unanswerable:
                    stats.unanswerable_queries += 1
                else:
                    stats.answerable_queries += 1
            for record in doc_evidence:
                stats.evidence_items += len(record.gold_evidence_texts)
                stats.matched_evidence_items += sum(
                    1 for match in record.evidence_matches if match.get("block_id")
                )
            stats.warnings.extend(doc.conversion_warnings)

        return documents, evidence_records, stats

    def convert_row(self, row: pd.Series, split: str) -> tuple[GovernedDocument, list[GoldEvidenceRecord]]:
        source_doc_id = normalize_text(row["id"])
        safe_doc = safe_id(source_doc_id)
        doc_id = f"qasper_{split}_{safe_doc}"
        title = normalize_text(row.get("title"))
        abstract = normalize_text(row.get("abstract"))

        full_text = parse_hf_csv_cell(row.get("full_text")) or {}
        qas = parse_hf_csv_cell(row.get("qas")) or {}
        figures_and_tables = parse_hf_csv_cell(row.get("figures_and_tables")) or {}

        warnings: list[str] = []
        blocks = self._build_blocks(
            doc_id=doc_id,
            source_doc_id=source_doc_id,
            split=split,
            title=title,
            abstract=abstract,
            full_text=full_text,
            figures_and_tables=figures_and_tables,
            warnings=warnings,
        )
        block_lookup = {block.block_id: block for block in blocks}
        queries, evidence_records = self._build_queries(
            doc_id=doc_id,
            source_doc_id=source_doc_id,
            split=split,
            qas=qas,
            blocks=blocks,
            block_lookup=block_lookup,
            warnings=warnings,
        )

        doc = GovernedDocument(
            doc_id=doc_id,
            dataset=self.dataset_name,
            split=split,
            source_doc_id=source_doc_id,
            title=title,
            abstract=abstract,
            normalization_status="provided_by_dataset",
            term_policy="dataset_provided",
            blocks=blocks,
            queries=queries,
            source_ref={
                "zip_path": str(self.zip_path),
                "csv_split": f"{split}.csv",
                "paper_id": source_doc_id,
            },
            conversion_warnings=warnings,
            metadata={
                "adapter": "QasperAdapter",
                "schema_version": "hsc-govdoc-v1",
                "block_count": len(blocks),
                "query_count": len(queries),
                "has_figures_and_tables": bool(figures_and_tables),
            },
        )
        return doc, evidence_records

    def _build_blocks(
        self,
        doc_id: str,
        source_doc_id: str,
        split: str,
        title: str,
        abstract: str,
        full_text: dict[str, Any],
        figures_and_tables: dict[str, Any],
        warnings: list[str],
    ) -> list[GovernedBlock]:
        blocks: list[GovernedBlock] = []
        order = 0
        heading_ids: dict[tuple[str, ...], str] = {}

        def next_id(kind: str) -> str:
            nonlocal order
            order += 1
            return f"{doc_id}_{kind}_{order:05d}"

        if title:
            block_id = next_id("title")
            blocks.append(
                GovernedBlock(
                    block_id=block_id,
                    doc_id=doc_id,
                    type="title",
                    text=title,
                    order=order,
                    level=0,
                    title_path=[title],
                    source_anchor=SourceAnchor(
                        dataset=self.dataset_name,
                        split=split,
                        source_doc_id=source_doc_id,
                        section_name="Title",
                    ),
                    metadata={"token_estimate": token_estimate(title)},
                )
            )

        if abstract:
            heading_id = next_id("heading")
            heading_ids[("Abstract",)] = heading_id
            blocks.append(
                GovernedBlock(
                    block_id=heading_id,
                    doc_id=doc_id,
                    type="heading",
                    text="Abstract",
                    order=order,
                    level=1,
                    title_path=["Abstract"],
                    source_anchor=SourceAnchor(
                        dataset=self.dataset_name,
                        split=split,
                        source_doc_id=source_doc_id,
                        section_name="Abstract",
                    ),
                    metadata={"generated_by_adapter": True},
                )
            )
            block_id = next_id("abstract")
            blocks.append(
                GovernedBlock(
                    block_id=block_id,
                    doc_id=doc_id,
                    type="abstract",
                    text=abstract,
                    order=order,
                    level=1,
                    title_path=["Abstract"],
                    parent_heading_id=heading_id,
                    source_anchor=SourceAnchor(
                        dataset=self.dataset_name,
                        split=split,
                        source_doc_id=source_doc_id,
                        section_name="Abstract",
                        paragraph_index=0,
                    ),
                    metadata={"token_estimate": token_estimate(abstract)},
                )
            )

        section_names = full_text.get("section_name") or []
        paragraphs_by_section = full_text.get("paragraphs") or []
        if len(section_names) != len(paragraphs_by_section):
            warnings.append(
                f"{doc_id}: full_text section_name/paragraphs length mismatch "
                f"({len(section_names)} vs {len(paragraphs_by_section)})"
            )

        for section_index, section_name_raw in enumerate(section_names):
            section_name = normalize_text(section_name_raw)
            if not section_name:
                continue
            title_path = [normalize_text(part) for part in section_name.split(":::")]
            title_path = [part for part in title_path if part]
            if not title_path:
                title_path = [section_name]

            parent_heading_id = None
            for depth in range(1, len(title_path) + 1):
                key = tuple(title_path[:depth])
                if key not in heading_ids:
                    heading_block_id = next_id("heading")
                    heading_ids[key] = heading_block_id
                    blocks.append(
                        GovernedBlock(
                            block_id=heading_block_id,
                            doc_id=doc_id,
                            type="heading",
                            text=title_path[depth - 1],
                            order=order,
                            level=depth,
                            title_path=title_path[:depth],
                            source_anchor=SourceAnchor(
                                dataset=self.dataset_name,
                                split=split,
                                source_doc_id=source_doc_id,
                                section_name=" ::: ".join(title_path[:depth]),
                            ),
                            parent_heading_id=parent_heading_id,
                            metadata={"generated_by_adapter": True},
                        )
                    )
                parent_heading_id = heading_ids[key]

            paragraphs = (
                paragraphs_by_section[section_index]
                if section_index < len(paragraphs_by_section)
                else []
            )
            if paragraphs is None:
                paragraphs = []
            if isinstance(paragraphs, str):
                paragraphs = [paragraphs]

            for para_index, para in enumerate(paragraphs):
                text = normalize_text(para)
                if not text:
                    continue
                block_id = next_id("paragraph")
                blocks.append(
                    GovernedBlock(
                        block_id=block_id,
                        doc_id=doc_id,
                        type="paragraph",
                        text=text,
                        order=order,
                        level=len(title_path),
                        title_path=title_path,
                        parent_heading_id=parent_heading_id,
                        source_anchor=SourceAnchor(
                            dataset=self.dataset_name,
                            split=split,
                            source_doc_id=source_doc_id,
                            section_name=section_name,
                            paragraph_index=para_index,
                        ),
                        metadata={
                            "section_index": section_index,
                            "paragraph_index": para_index,
                            "token_estimate": token_estimate(text),
                        },
                    )
                )

        caption_list = figures_and_tables.get("caption") or []
        file_list = figures_and_tables.get("file") or []
        for asset_index, caption_raw in enumerate(caption_list):
            caption = normalize_text(caption_raw)
            if not caption:
                continue
            file_name = normalize_text(file_list[asset_index]) if asset_index < len(file_list) else ""
            lower = caption.lower()
            if lower.startswith("table"):
                block_type = "table"
            elif lower.startswith("figure"):
                block_type = "figure"
            else:
                block_type = "caption"
            block_id = next_id(block_type)
            blocks.append(
                GovernedBlock(
                    block_id=block_id,
                    doc_id=doc_id,
                    type=block_type,
                    text=caption,
                    order=order,
                    level=1,
                    title_path=["Figures and Tables"],
                    source_anchor=SourceAnchor(
                        dataset=self.dataset_name,
                        split=split,
                        source_doc_id=source_doc_id,
                        section_name="Figures and Tables",
                        paragraph_index=asset_index,
                        asset_file=file_name or None,
                    ),
                    metadata={
                        "protected_block": True,
                        "asset_file": file_name,
                        "token_estimate": token_estimate(caption),
                    },
                )
            )

        return blocks

    def _build_queries(
        self,
        doc_id: str,
        source_doc_id: str,
        split: str,
        qas: dict[str, Any],
        blocks: list[GovernedBlock],
        block_lookup: dict[str, GovernedBlock],
        warnings: list[str],
    ) -> tuple[list[GovernedQuery], list[GoldEvidenceRecord]]:
        questions = qas.get("question") or []
        question_ids = qas.get("question_id") or []
        answers_list = qas.get("answers") or []
        nlp_background = qas.get("nlp_background") or []
        topic_background = qas.get("topic_background") or []
        paper_read = qas.get("paper_read") or []

        queries: list[GovernedQuery] = []
        evidence_records: list[GoldEvidenceRecord] = []

        for index, question_raw in enumerate(questions):
            question = normalize_text(question_raw)
            if not question:
                continue
            source_question_id = (
                normalize_text(question_ids[index])
                if index < len(question_ids)
                else stable_hash(question)
            )
            query_id = f"{doc_id}_q_{safe_id(source_question_id[:24])}"
            raw_answers = answers_list[index] if index < len(answers_list) else None
            answer_bundle = self._extract_answer_bundle(raw_answers)
            evidence_matches = [
                self._match_evidence_to_block(evidence, blocks)
                for evidence in answer_bundle["evidence_texts"]
            ]
            gold_block_ids = []
            for match in evidence_matches:
                if match.block_id and match.block_id not in gold_block_ids:
                    gold_block_ids.append(match.block_id)

            if answer_bundle["evidence_texts"] and not gold_block_ids:
                warnings.append(
                    f"{doc_id}/{query_id}: no evidence block matched for "
                    f"{len(answer_bundle['evidence_texts'])} evidence item(s)"
                )

            scores = [m.score for m in evidence_matches if m.block_id]
            avg_score = sum(scores) / len(scores) if scores else None
            difficulty = self._build_difficulty(
                nlp_background[index] if index < len(nlp_background) else None,
                topic_background[index] if index < len(topic_background) else None,
                paper_read[index] if index < len(paper_read) else None,
            )
            query = GovernedQuery(
                query_id=query_id,
                doc_id=doc_id,
                dataset=self.dataset_name,
                split=split,
                question=question,
                answer=answer_bundle["answer_text"],
                answer_type=answer_bundle["answer_type"],
                is_unanswerable=answer_bundle["is_unanswerable"],
                gold_block_ids=gold_block_ids,
                gold_evidence_texts=answer_bundle["evidence_texts"],
                evidence_match_score=avg_score,
                difficulty=difficulty,
                source_question_id=source_question_id,
                metadata={
                    "nlp_background": nlp_background[index] if index < len(nlp_background) else None,
                    "topic_background": topic_background[index] if index < len(topic_background) else None,
                    "paper_read": paper_read[index] if index < len(paper_read) else None,
                    "answer_count": answer_bundle["answer_count"],
                },
            )
            queries.append(query)
            evidence_records.append(
                GoldEvidenceRecord(
                    query_id=query_id,
                    doc_id=doc_id,
                    dataset=self.dataset_name,
                    split=split,
                    question=question,
                    answer=answer_bundle["answer_text"],
                    gold_block_ids=gold_block_ids,
                    gold_evidence_texts=answer_bundle["evidence_texts"],
                    evidence_matches=[
                        {
                            "evidence_text": match.evidence_text,
                            "block_id": match.block_id,
                            "score": match.score,
                            "method": match.method,
                            "block_type": block_lookup[match.block_id].type
                            if match.block_id in block_lookup
                            else None,
                        }
                        for match in evidence_matches
                    ],
                    is_unanswerable=answer_bundle["is_unanswerable"],
                )
            )

        return queries, evidence_records

    def _extract_answer_bundle(self, raw_answers: Any) -> dict[str, Any]:
        if not raw_answers:
            return {
                "answer_text": "",
                "answer_type": "missing",
                "is_unanswerable": True,
                "evidence_texts": [],
                "answer_count": 0,
            }

        answer_items: list[dict[str, Any]] = []
        if isinstance(raw_answers, dict) and isinstance(raw_answers.get("answer"), list):
            answer_items = raw_answers.get("answer") or []
        elif isinstance(raw_answers, list):
            answer_items = raw_answers
        elif isinstance(raw_answers, dict):
            answer_items = [raw_answers]

        answer_texts: list[str] = []
        evidence_texts: list[str] = []
        is_unanswerable = False
        answer_type = "unknown"

        for item in answer_items:
            if not isinstance(item, dict):
                continue
            if item.get("unanswerable"):
                is_unanswerable = True
                answer_type = "unanswerable"

            free_form = normalize_text(item.get("free_form_answer"))
            spans = [normalize_text(x) for x in item.get("extractive_spans") or [] if normalize_text(x)]
            yes_no = item.get("yes_no")
            if free_form:
                answer_texts.append(free_form)
                answer_type = "free_form"
            elif spans:
                answer_texts.extend(spans)
                answer_type = "extractive"
            elif yes_no is not None:
                answer_texts.append(str(yes_no))
                answer_type = "yes_no"

            for field in ("evidence", "highlighted_evidence"):
                for evidence in item.get(field) or []:
                    normalized = normalize_text(evidence)
                    if normalized and normalized not in evidence_texts:
                        evidence_texts.append(normalized)

        if not answer_texts and is_unanswerable:
            answer_texts.append("UNANSWERABLE")
        return {
            "answer_text": " | ".join(dict.fromkeys(answer_texts)),
            "answer_type": answer_type,
            "is_unanswerable": is_unanswerable,
            "evidence_texts": evidence_texts,
            "answer_count": len(answer_items),
        }

    def _match_evidence_to_block(
        self,
        evidence: str,
        blocks: list[GovernedBlock],
    ) -> EvidenceMatch:
        evidence_norm = normalize_for_match(evidence)
        if not evidence_norm:
            return EvidenceMatch(evidence_text=evidence, block_id=None, score=0.0, method="empty")

        best_block_id: str | None = None
        best_score = 0.0
        best_method = "token_overlap"

        for block in blocks:
            if block.type not in {"abstract", "paragraph", "table", "figure", "caption"}:
                continue
            block_norm = normalize_for_match(block.text)
            if not block_norm:
                continue
            if evidence_norm in block_norm:
                coverage = len(evidence_norm) / max(len(block_norm), 1)
                score = min(1.0, 0.85 + coverage)
                if score > best_score:
                    best_block_id = block.block_id
                    best_score = score
                    best_method = "substring"
                continue
            if block_norm in evidence_norm:
                coverage = len(block_norm) / max(len(evidence_norm), 1)
                score = min(0.95, 0.75 + coverage)
                if score > best_score:
                    best_block_id = block.block_id
                    best_score = score
                    best_method = "block_in_evidence"
                continue

            score = self._token_overlap_score(evidence_norm, block_norm)
            if score > best_score:
                best_block_id = block.block_id
                best_score = score

        if best_score < 0.18:
            return EvidenceMatch(
                evidence_text=evidence,
                block_id=None,
                score=best_score,
                method="unmatched",
            )
        return EvidenceMatch(
            evidence_text=evidence,
            block_id=best_block_id,
            score=round(best_score, 4),
            method=best_method,
        )

    def _token_overlap_score(self, evidence_norm: str, block_norm: str) -> float:
        evidence_tokens = set(evidence_norm.split())
        block_tokens = set(block_norm.split())
        if not evidence_tokens or not block_tokens:
            return 0.0
        intersection = evidence_tokens & block_tokens
        recall = len(intersection) / len(evidence_tokens)
        precision = len(intersection) / len(block_tokens)
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    def _build_difficulty(self, nlp: Any, topic: Any, paper_read: Any) -> str:
        parts = [
            f"nlp={normalize_text(nlp) or 'unknown'}",
            f"topic={normalize_text(topic) or 'unknown'}",
            f"paper_read={normalize_text(paper_read) or 'unknown'}",
        ]
        return ";".join(parts)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query in queries:
            row = query.model_dump()
            row["gold_block_ids"] = json.dumps(row["gold_block_ids"], ensure_ascii=False)
            row["gold_evidence_texts"] = json.dumps(row["gold_evidence_texts"], ensure_ascii=False)
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            writer.writerow(row)
