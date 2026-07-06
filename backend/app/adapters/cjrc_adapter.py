# -*- coding: utf-8 -*-
"""CJRC/CAIL2019 reading-comprehension adapter.

The CAIL2019 reading-comprehension task is commonly used as the public CJRC
dataset. Its records contain legal case contexts, case names, questions, answer
spans, and impossible-question flags. This adapter maps those records into the
HSC-RAG GovernedDocument contract and derives block-level gold evidence from
answer_start offsets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
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
STRONG_BOUNDARIES = set("。！？!?；;\n")
WEAK_BOUNDARIES = set("，,、：:")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def safe_id(value: Any) -> str:
    raw = normalize_text(value)
    cleaned = NON_ID_RE.sub("_", raw).strip("_")
    return cleaned or stable_hash(raw or "cjrc")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


@dataclass(frozen=True)
class BlockSpan:
    block: GovernedBlock
    paragraph_index: int
    char_start: int
    char_end: int


@dataclass
class CjrcConversionStats:
    split: str
    source_path: str
    documents: int = 0
    blocks: int = 0
    queries: int = 0
    answerable_queries: int = 0
    unanswerable_queries: int = 0
    spanless_answerable_queries: int = 0
    evidence_items: int = 0
    matched_evidence_items: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": "cjrc_cail2019",
            "split": self.split,
            "source_path": self.source_path,
            "documents": self.documents,
            "blocks": self.blocks,
            "queries": self.queries,
            "answerable_queries": self.answerable_queries,
            "unanswerable_queries": self.unanswerable_queries,
            "spanless_answerable_queries": self.spanless_answerable_queries,
            "evidence_items": self.evidence_items,
            "matched_evidence_items": self.matched_evidence_items,
            "evidence_match_rate": (
                self.matched_evidence_items / self.evidence_items
                if self.evidence_items
                else None
            ),
            "warnings": self.warnings[:200],
        }


class CjrcAdapter:
    """Convert CAIL2019/CJRC reading-comprehension JSON to governed artifacts."""

    dataset_name = "cjrc_cail2019"

    def __init__(
        self,
        input_path: str | Path,
        *,
        block_mode: str = "sentence",
        max_block_chars: int = 260,
    ):
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise FileNotFoundError(f"CJRC file not found: {self.input_path}")
        if block_mode not in {"sentence", "paragraph"}:
            raise ValueError("block_mode must be one of: sentence, paragraph")
        self.block_mode = block_mode
        self.max_block_chars = max(80, int(max_block_chars))

    def load_payload(self) -> dict[str, Any]:
        with self.input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("CJRC input must be a JSON object.")
        if not isinstance(payload.get("data"), list):
            raise ValueError("CJRC input must contain a list field named 'data'.")
        return payload

    def convert(
        self,
        *,
        split: str,
        limit_docs: int | None = None,
    ) -> tuple[list[GovernedDocument], list[GoldEvidenceRecord], CjrcConversionStats]:
        payload = self.load_payload()
        stats = CjrcConversionStats(split=split, source_path=str(self.input_path))
        docs: list[GovernedDocument] = []
        evidence: list[GoldEvidenceRecord] = []

        for case_index, case in enumerate(payload.get("data", [])):
            if limit_docs is not None and len(docs) >= limit_docs:
                break
            if not isinstance(case, dict):
                stats.warnings.append(f"case={case_index}: skipped non-object case")
                continue
            try:
                doc, doc_evidence = self.convert_case(case=case, case_index=case_index, split=split)
            except Exception as exc:
                stats.warnings.append(
                    f"case={case_index}: conversion failed: {type(exc).__name__}: {exc}"
                )
                continue

            docs.append(doc)
            evidence.extend(doc_evidence)
            stats.documents += 1
            stats.blocks += len(doc.blocks)
            stats.queries += len(doc.queries)
            stats.answerable_queries += sum(1 for query in doc.queries if not query.is_unanswerable)
            stats.unanswerable_queries += sum(1 for query in doc.queries if query.is_unanswerable)
            stats.spanless_answerable_queries += sum(
                1
                for query in doc.queries
                if query.metadata.get("original_is_impossible") is False
                and query.metadata.get("has_answer_text") is True
                and not query.gold_block_ids
            )
            stats.evidence_items += sum(
                1
                for record in doc_evidence
                if record.answer and not record.is_unanswerable
            )
            stats.matched_evidence_items += sum(
                1
                for record in doc_evidence
                if record.answer and not record.is_unanswerable and record.gold_block_ids
            )
            stats.warnings.extend(doc.conversion_warnings)

        return docs, evidence, stats

    def convert_case(
        self,
        *,
        case: dict[str, Any],
        case_index: int,
        split: str,
    ) -> tuple[GovernedDocument, list[GoldEvidenceRecord]]:
        caseid = normalize_text(case.get("caseid")) or f"{split}_{case_index:05d}"
        domain = normalize_text(case.get("domain")) or "unknown"
        paragraphs = case.get("paragraphs") or []
        if not isinstance(paragraphs, list):
            paragraphs = []

        casename = self._case_name(paragraphs) or f"CJRC case {caseid}"
        doc_id = f"cjrc_{split}_{safe_id(caseid)[:32]}"
        warnings: list[str] = []
        blocks: list[GovernedBlock] = []
        spans: list[BlockSpan] = []
        order = 1

        blocks.append(
            GovernedBlock(
                block_id=f"{doc_id}_title_00001",
                doc_id=doc_id,
                type="title",
                text=casename,
                order=order,
                level=0,
                title_path=[casename],
                source_anchor=SourceAnchor(
                    dataset=self.dataset_name,
                    split=split,
                    source_doc_id=caseid,
                    section_name="case_title",
                    extra={"caseid": caseid, "domain": domain, "case_index": case_index},
                ),
                entity_tags=[casename, domain],
                metadata={"generated_by_adapter": True, "caseid": caseid, "domain": domain},
            )
        )
        order += 1

        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                warnings.append(f"{doc_id}: paragraph={paragraph_index} skipped non-object paragraph")
                continue
            context = str(paragraph.get("context") or "")
            paragraph_casename = normalize_text(paragraph.get("casename")) or casename
            section_name = f"case_paragraph_{paragraph_index + 1}"
            heading_id = f"{doc_id}_heading_{order:05d}"
            blocks.append(
                GovernedBlock(
                    block_id=heading_id,
                    doc_id=doc_id,
                    type="heading",
                    text=section_name,
                    order=order,
                    level=1,
                    title_path=[casename, section_name],
                    source_anchor=SourceAnchor(
                        dataset=self.dataset_name,
                        split=split,
                        source_doc_id=caseid,
                        section_name=section_name,
                        paragraph_index=paragraph_index,
                        extra={
                            "caseid": caseid,
                            "domain": domain,
                            "casename": paragraph_casename,
                            "case_index": case_index,
                        },
                    ),
                    parent_heading_id=None,
                    entity_tags=[paragraph_casename, domain],
                    metadata={"generated_by_adapter": True, "paragraph_index": paragraph_index},
                )
            )
            order += 1

            for span_index, (start, end) in enumerate(self._content_spans(context)):
                text = normalize_text(context[start:end])
                if not text:
                    continue
                block_id = f"{doc_id}_p{paragraph_index:03d}_b{span_index:04d}"
                block = GovernedBlock(
                    block_id=block_id,
                    doc_id=doc_id,
                    type="paragraph",
                    text=text,
                    order=order,
                    level=1,
                    title_path=[casename, section_name],
                    parent_heading_id=heading_id,
                    source_anchor=SourceAnchor(
                        dataset=self.dataset_name,
                        split=split,
                        source_doc_id=caseid,
                        section_name=section_name,
                        paragraph_index=paragraph_index,
                        extra={
                            "caseid": caseid,
                            "domain": domain,
                            "casename": paragraph_casename,
                            "case_index": case_index,
                            "context_char_start": start,
                            "context_char_end": end,
                            "block_mode": self.block_mode,
                        },
                    ),
                    entity_tags=[paragraph_casename, domain],
                    metadata={
                        "caseid": caseid,
                        "domain": domain,
                        "casename": paragraph_casename,
                        "paragraph_index": paragraph_index,
                        "context_char_start": start,
                        "context_char_end": end,
                        "block_mode": self.block_mode,
                    },
                )
                blocks.append(block)
                spans.append(
                    BlockSpan(
                        block=block,
                        paragraph_index=paragraph_index,
                        char_start=start,
                        char_end=end,
                    )
                )
                order += 1

        queries, evidence = self._build_queries(
            doc_id=doc_id,
            caseid=caseid,
            casename=casename,
            domain=domain,
            split=split,
            paragraphs=paragraphs,
            spans=spans,
            warnings=warnings,
        )
        doc = GovernedDocument(
            doc_id=doc_id,
            dataset=self.dataset_name,
            split=split,
            source_doc_id=caseid,
            title=casename,
            normalization_status="provided_by_dataset",
            term_policy="cjrc_cail2019_span_annotations",
            governance_stage="post_normalization_packaging",
            blocks=blocks,
            queries=queries,
            source_ref={
                "input_path": str(self.input_path),
                "caseid": caseid,
                "domain": domain,
                "case_index": case_index,
            },
            conversion_warnings=warnings,
            metadata={
                "adapter": "CjrcAdapter",
                "schema_version": "hsc-govdoc-v1",
                "block_mode": self.block_mode,
                "max_block_chars": self.max_block_chars,
                "caseid": caseid,
                "domain": domain,
                "paragraph_count": len(paragraphs),
                "query_count": len(queries),
            },
        )
        return doc, evidence

    def _case_name(self, paragraphs: list[Any]) -> str:
        for paragraph in paragraphs:
            if isinstance(paragraph, dict):
                casename = normalize_text(paragraph.get("casename"))
                if casename:
                    return casename
        return ""

    def _content_spans(self, context: str) -> list[tuple[int, int]]:
        if self.block_mode == "paragraph":
            return self._split_long_span(context, *self._trim_span(context, 0, len(context)))

        strong: list[tuple[int, int]] = []
        start = 0
        for index, char in enumerate(context):
            if char in STRONG_BOUNDARIES:
                span = self._trim_span(context, start, index + 1)
                if span[0] < span[1]:
                    strong.append(span)
                start = index + 1
        final_span = self._trim_span(context, start, len(context))
        if final_span[0] < final_span[1]:
            strong.append(final_span)

        spans: list[tuple[int, int]] = []
        for span in strong:
            spans.extend(self._split_long_span(context, *span))
        return spans

    def _split_long_span(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        if start >= end:
            return []
        if end - start <= self.max_block_chars:
            return [(start, end)]

        spans: list[tuple[int, int]] = []
        cursor = start
        last_boundary = start
        index = start
        while index < end:
            if text[index] in WEAK_BOUNDARIES:
                last_boundary = index + 1
            if index - cursor + 1 >= self.max_block_chars:
                split_at = last_boundary if last_boundary > cursor else index + 1
                span = self._trim_span(text, cursor, split_at)
                if span[0] < span[1]:
                    spans.append(span)
                cursor = split_at
                last_boundary = cursor
            index += 1
        final_span = self._trim_span(text, cursor, end)
        if final_span[0] < final_span[1]:
            spans.append(final_span)
        return spans

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _build_queries(
        self,
        *,
        doc_id: str,
        caseid: str,
        casename: str,
        domain: str,
        split: str,
        paragraphs: list[Any],
        spans: list[BlockSpan],
        warnings: list[str],
    ) -> tuple[list[GovernedQuery], list[GoldEvidenceRecord]]:
        queries: list[GovernedQuery] = []
        evidence_records: list[GoldEvidenceRecord] = []

        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue
            context = str(paragraph.get("context") or "")
            qas = paragraph.get("qas") or []
            if not isinstance(qas, list):
                warnings.append(f"{doc_id}: paragraph={paragraph_index} qas is not a list")
                continue
            paragraph_spans = [span for span in spans if span.paragraph_index == paragraph_index]
            for qa_index, qa in enumerate(qas):
                if not isinstance(qa, dict):
                    continue
                question = normalize_text(qa.get("question"))
                if not question:
                    continue
                source_question_id = normalize_text(qa.get("id")) or f"{caseid}_p{paragraph_index}_q{qa_index}"
                query_id = f"{doc_id}_q_{safe_id(source_question_id)[:32]}"
                original_is_impossible = parse_bool(qa.get("is_impossible"))
                answers = self._extract_answers(qa)
                gold_blocks, evidence_matches = self._match_answers(
                    answers=answers,
                    context=context,
                    spans=paragraph_spans,
                    warnings=warnings,
                    query_id=query_id,
                )
                answer_text = " | ".join(dict.fromkeys(answer["text"] for answer in answers if answer["text"]))
                has_answer_text = bool(answer_text)
                answer_type = self._answer_type(answers, original_is_impossible, bool(gold_blocks))
                is_unanswerable = original_is_impossible or not gold_blocks
                gold_block_ids = [block.block_id for block in gold_blocks]
                gold_evidence_texts = [block.text for block in gold_blocks]
                match_scores = [
                    float(item.get("score", 0.0))
                    for item in evidence_matches
                    if item.get("block_id")
                ]
                query = GovernedQuery(
                    query_id=query_id,
                    doc_id=doc_id,
                    dataset=self.dataset_name,
                    split=split,
                    question=question,
                    answer=answer_text,
                    answer_type=answer_type,
                    is_unanswerable=is_unanswerable,
                    gold_block_ids=gold_block_ids,
                    gold_evidence_texts=gold_evidence_texts,
                    evidence_match_score=(
                        round(sum(match_scores) / len(match_scores), 4)
                        if match_scores
                        else None
                    ),
                    question_type="cjrc_legal_span_qa",
                    source_question_id=source_question_id,
                    metadata={
                        "caseid": caseid,
                        "casename": casename,
                        "domain": domain,
                        "paragraph_index": paragraph_index,
                        "qa_index": qa_index,
                        "original_is_impossible": original_is_impossible,
                        "has_answer_text": has_answer_text,
                        "answer_count": len(answers),
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
                        answer=answer_text,
                        gold_block_ids=gold_block_ids,
                        gold_evidence_texts=gold_evidence_texts,
                        evidence_matches=evidence_matches,
                        is_unanswerable=is_unanswerable,
                    )
                )

        return queries, evidence_records

    def _extract_answers(self, qa: dict[str, Any]) -> list[dict[str, Any]]:
        answers = qa.get("answers") or []
        if not isinstance(answers, list):
            return []
        extracted: list[dict[str, Any]] = []
        for answer in answers:
            if not isinstance(answer, dict):
                continue
            text = normalize_text(answer.get("text"))
            if not text:
                continue
            raw_start = answer.get("answer_start")
            start = raw_start if isinstance(raw_start, int) else None
            extracted.append({"text": text, "answer_start": start})
        return extracted

    def _match_answers(
        self,
        *,
        answers: list[dict[str, Any]],
        context: str,
        spans: list[BlockSpan],
        warnings: list[str],
        query_id: str,
    ) -> tuple[list[GovernedBlock], list[dict[str, Any]]]:
        gold_blocks: list[GovernedBlock] = []
        evidence_matches: list[dict[str, Any]] = []

        def add_block(block: GovernedBlock) -> None:
            if block.block_id not in {item.block_id for item in gold_blocks}:
                gold_blocks.append(block)

        for answer in answers:
            answer_text = answer["text"]
            start = answer.get("answer_start")
            method = "answer_start_overlap"
            if not isinstance(start, int) or start < 0:
                found = context.find(answer_text)
                start = found if found >= 0 else None
                method = "answer_text_substring"

            matched_blocks: list[GovernedBlock] = []
            if isinstance(start, int):
                end = start + len(answer_text)
                for span in spans:
                    if start < span.char_end and end > span.char_start:
                        matched_blocks.append(span.block)
                        add_block(span.block)

            if not matched_blocks:
                for span in spans:
                    if answer_text and answer_text in span.block.text:
                        matched_blocks.append(span.block)
                        add_block(span.block)
                if matched_blocks:
                    method = "block_text_substring"

            if matched_blocks:
                for block in matched_blocks:
                    evidence_matches.append(
                        {
                            "answer_text": answer_text,
                            "answer_start": start,
                            "block_id": block.block_id,
                            "score": 1.0 if method == "answer_start_overlap" else 0.85,
                            "method": method,
                            "block_text": block.text,
                        }
                    )
            else:
                evidence_matches.append(
                    {
                        "answer_text": answer_text,
                        "answer_start": start,
                        "block_id": None,
                        "score": 0.0,
                        "method": "unmatched",
                    }
                )
                if answer_text not in {"YES", "NO"}:
                    warnings.append(f"{query_id}: answer did not match any block: {answer_text[:80]}")

        return gold_blocks, evidence_matches

    def _answer_type(self, answers: list[dict[str, Any]], original_is_impossible: bool, has_gold: bool) -> str:
        if original_is_impossible:
            return "impossible"
        texts = {answer["text"] for answer in answers if answer["text"]}
        if texts and texts.issubset({"YES", "NO"}):
            return "yes_no_with_span" if has_gold else "yes_no_without_span"
        return "span_extraction" if has_gold else "span_without_evidence"


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
