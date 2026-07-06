# -*- coding: utf-8 -*-
"""LongBench-v2 -> GovernedDocument adapter.

LongBench-v2 provides long contexts, multiple-choice questions, choices and the
correct choice label, but it does not provide sentence-level gold evidence.
This adapter therefore treats evidence as optional weak evidence: it only emits
gold block ids when the correct choice text can be matched to context blocks by
an auditable generic policy.
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
from typing import Any, Literal

from app.core.schemas import (
    GoldEvidenceRecord,
    GovernedBlock,
    GovernedDocument,
    GovernedQuery,
    SourceAnchor,
)


EvidenceMode = Literal["none", "choice_exact", "choice_overlap"]

WHITESPACE_RE = re.compile(r"\s+")
NON_ID_RE = re.compile(r"[^0-9A-Za-z_.-]+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;。！？；])\s+")
HEADING_RE = re.compile(
    r"^(?:chapter|section|part|appendix|article|contents?|preface|abstract|"
    r"introduction|conclusion|references|[IVXLCDM]+\.|\d+(?:\.\d+)*\.?)\b",
    re.IGNORECASE,
)
LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_+\-]{2,}(?:\s+[A-Z][A-Za-z0-9_+\-]{2,}){0,3}\b")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def clean_context_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = CONTROL_RE.sub(" ", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    return value.strip()


def safe_id(value: str) -> str:
    value = NON_ID_RE.sub("_", value).strip("_")
    if value:
        return value[:96]
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]


def infer_member(zip_path: Path, member: str | None = None) -> str:
    if member:
        return member
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
    if not members:
        raise FileNotFoundError(f"No CSV member found in {zip_path}")
    preferred = [name for name in members if "longbench" in name.lower()]
    return preferred[0] if preferred else members[0]


@dataclass
class LongBenchConversionStats:
    split: str
    evidence_mode: EvidenceMode
    documents: int = 0
    blocks: int = 0
    queries: int = 0
    answerable_queries: int = 0
    weak_evidence_queries: int = 0
    evidence_items: int = 0
    exact_choice_matches: int = 0
    overlap_choice_matches: int = 0
    json_contexts: int = 0
    truncated_documents: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": "longbench_v2",
            "split": self.split,
            "documents": self.documents,
            "blocks": self.blocks,
            "queries": self.queries,
            "answerable_queries_for_retrieval_eval": self.answerable_queries,
            "weak_evidence_queries": self.weak_evidence_queries,
            "evidence_items": self.evidence_items,
            "weak_evidence_query_rate": (
                self.weak_evidence_queries / self.queries if self.queries else None
            ),
            "evidence_mode": self.evidence_mode,
            "evidence_policy": (
                "LongBench-v2 has answer labels but no annotated evidence. "
                "Generated evidence is weak and derived only from generic correct-choice text matching."
            ),
            "exact_choice_matches": self.exact_choice_matches,
            "overlap_choice_matches": self.overlap_choice_matches,
            "json_contexts": self.json_contexts,
            "truncated_documents": self.truncated_documents,
            "warnings": self.warnings[:200],
        }


class LongBenchAdapter:
    """Convert LongBench-v2 CSV-in-zip artifacts into HSC-RAG contracts."""

    def __init__(
        self,
        zip_path: str | Path,
        member: str | None = None,
        *,
        max_block_chars: int = 1800,
        max_blocks_per_doc: int | None = None,
        evidence_mode: EvidenceMode = "choice_overlap",
    ):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"LongBench zip not found: {self.zip_path}")
        if max_block_chars < 200:
            raise ValueError("max_block_chars must be at least 200")
        if evidence_mode not in {"none", "choice_exact", "choice_overlap"}:
            raise ValueError(f"Unknown evidence mode: {evidence_mode}")
        self.member = infer_member(self.zip_path, member)
        self.max_block_chars = max_block_chars
        self.max_blocks_per_doc = max_blocks_per_doc if max_blocks_per_doc and max_blocks_per_doc > 0 else None
        self.evidence_mode: EvidenceMode = evidence_mode

    def iter_records(self, *, limit_docs: int | None = None) -> Iterable[dict[str, str]]:
        csv.field_size_limit(2**31 - 1)
        yielded = 0
        with zipfile.ZipFile(self.zip_path) as zf:
            with zf.open(self.member) as raw:
                stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                reader = csv.DictReader(stream)
                for row in reader:
                    yield row
                    yielded += 1
                    if limit_docs is not None and yielded >= limit_docs:
                        return

    def convert(
        self,
        *,
        split: str = "eval",
        limit_docs: int | None = 20,
    ) -> tuple[list[GovernedDocument], list[GoldEvidenceRecord], LongBenchConversionStats]:
        stats = LongBenchConversionStats(split=split, evidence_mode=self.evidence_mode)
        docs: list[GovernedDocument] = []
        evidence: list[GoldEvidenceRecord] = []

        for row_index, record in enumerate(self.iter_records(limit_docs=limit_docs)):
            try:
                doc, gold = self.convert_record(record, split=split, row_index=row_index)
            except Exception as exc:
                stats.warnings.append(f"row={row_index}: conversion failed: {type(exc).__name__}: {exc}")
                continue

            docs.append(doc)
            evidence.extend(gold)
            stats.documents += 1
            stats.blocks += len(doc.blocks)
            stats.queries += len(doc.queries)
            stats.answerable_queries += sum(1 for query in doc.queries if not query.is_unanswerable)
            stats.weak_evidence_queries += sum(
                1
                for query in doc.queries
                if query.metadata.get("evidence_policy") != "none" and query.gold_block_ids
            )
            stats.evidence_items += sum(len(item.gold_evidence_texts) for item in gold)
            stats.exact_choice_matches += sum(
                1
                for item in gold
                for match in item.evidence_matches
                if match.get("method") == "longbench_choice_exact"
            )
            stats.overlap_choice_matches += sum(
                1
                for item in gold
                for match in item.evidence_matches
                if match.get("method") == "longbench_choice_overlap"
            )
            if doc.metadata.get("context_format") == "json":
                stats.json_contexts += 1
            if doc.metadata.get("truncated_blocks"):
                stats.truncated_documents += 1
            stats.warnings.extend(doc.conversion_warnings)

        return docs, evidence, stats

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        split: str,
        row_index: int = 0,
    ) -> tuple[GovernedDocument, list[GoldEvidenceRecord]]:
        raw_id = normalize_text(record.get("_id")) or f"row_{row_index}"
        doc_id = f"longbench_{split}_{safe_id(raw_id)}"
        domain = normalize_text(record.get("domain")) or "LongBench"
        sub_domain = normalize_text(record.get("sub_domain")) or "unknown"
        difficulty = normalize_text(record.get("difficulty")) or None
        length_label = normalize_text(record.get("length")) or None
        question = normalize_text(record.get("question"))
        answer_label = normalize_answer_label(record.get("answer"))
        choices = {
            label: normalize_text(record.get(f"choice_{label}"))
            for label in ["A", "B", "C", "D"]
        }
        answer_text = choices.get(answer_label, "") if answer_label else ""
        answer = f"{answer_label}: {answer_text}" if answer_label and answer_text else normalize_text(record.get("answer"))
        warnings: list[str] = []

        context = clean_context_text(record.get("context"))
        sections, context_format = context_sections(context)
        blocks: list[GovernedBlock] = []
        order = 1
        truncated = False

        for section_index, (section_path, section_text) in enumerate(sections):
            if not section_text:
                continue
            base_title_path = [domain, sub_domain, *section_path]
            new_blocks, order = self._blocks_from_text(
                text=section_text,
                doc_id=doc_id,
                raw_id=raw_id,
                split=split,
                base_title_path=base_title_path,
                section_index=section_index,
                start_order=order,
            )
            blocks.extend(new_blocks)
            if self.max_blocks_per_doc and len(blocks) >= self.max_blocks_per_doc:
                blocks = blocks[: self.max_blocks_per_doc]
                truncated = True
                warnings.append(f"document truncated at max_blocks_per_doc={self.max_blocks_per_doc}")
                break

        if not blocks:
            warnings.append("context produced no blocks")

        gold_blocks, evidence_matches = self._derive_evidence(
            blocks=blocks,
            choice_text=answer_text,
            question=question,
        )
        evidence_policy = (
            "none"
            if self.evidence_mode == "none"
            else "choice_text_weak_match"
        )
        retrieval_evaluable = bool(gold_blocks)
        query_id = f"{doc_id}_q"
        query = GovernedQuery(
            query_id=query_id,
            doc_id=doc_id,
            dataset="longbench_v2",
            split=split,
            question=question,
            answer=answer,
            answer_type="multiple_choice",
            is_unanswerable=not retrieval_evaluable,
            gold_block_ids=[block.block_id for block in gold_blocks],
            gold_evidence_texts=[block.text for block in gold_blocks],
            evidence_match_score=(
                max(float(match.get("score", 0.0)) for match in evidence_matches)
                if evidence_matches
                else None
            ),
            question_type="longbench_multiple_choice",
            difficulty=difficulty,
            source_question_id=raw_id,
            metadata={
                "domain": domain,
                "sub_domain": sub_domain,
                "length": length_label,
                "choices": choices,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "longbench_answerable": True,
                "retrieval_evaluable": retrieval_evaluable,
                "evidence_policy": evidence_policy,
                "evidence_is_weak": retrieval_evaluable,
            },
        )

        title = f"{domain} / {sub_domain} / {raw_id}"
        doc = GovernedDocument(
            doc_id=doc_id,
            dataset="longbench_v2",
            split=split,
            source_doc_id=raw_id,
            title=title,
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
                "adapter": "longbench_adapter",
                "domain": domain,
                "sub_domain": sub_domain,
                "difficulty": difficulty,
                "length": length_label,
                "context_format": context_format,
                "choices": choices,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "evidence_policy": evidence_policy,
                "evidence_is_weak": retrieval_evaluable,
                "truncated_blocks": truncated,
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
            evidence_matches=evidence_matches,
            is_unanswerable=query.is_unanswerable,
        )
        return doc, [gold_record]

    def _blocks_from_text(
        self,
        *,
        text: str,
        doc_id: str,
        raw_id: str,
        split: str,
        base_title_path: list[str],
        section_index: int,
        start_order: int,
    ) -> tuple[list[GovernedBlock], int]:
        blocks: list[GovernedBlock] = []
        order = start_order
        heading_path: list[str] = []
        paragraphs = split_paragraphs(text)

        for paragraph_index, paragraph in enumerate(paragraphs):
            pieces = split_long_block(paragraph, self.max_block_chars)
            for piece_index, piece in enumerate(pieces):
                clean_piece = piece.strip()
                if not clean_piece:
                    continue
                block_type = infer_block_type(clean_piece)
                if block_type == "heading":
                    heading = normalize_text(clean_piece)[:120]
                    heading_path = [heading]
                title_path = [*base_title_path, *heading_path]
                block_id = f"{doc_id}_b{order:06d}"
                blocks.append(
                    GovernedBlock(
                        block_id=block_id,
                        doc_id=doc_id,
                        type=block_type,
                        text=normalize_block_text(clean_piece, block_type),
                        order=order,
                        level=len(title_path),
                        title_path=title_path,
                        source_anchor=SourceAnchor(
                            dataset="longbench_v2",
                            split=split,
                            source_doc_id=raw_id,
                            section_name=" / ".join(title_path) if title_path else None,
                            paragraph_index=paragraph_index,
                            extra={
                                "section_index": section_index,
                                "piece_index": piece_index,
                                "block_type_inferred": block_type,
                            },
                        ),
                        entity_tags=derive_entities(" ".join(title_path), clean_piece),
                        metadata={
                            "section_index": section_index,
                            "paragraph_index": paragraph_index,
                            "piece_index": piece_index,
                            "base_title_path": base_title_path,
                            "inferred_block_type": block_type,
                        },
                    )
                )
                order += 1
        return blocks, order

    def _derive_evidence(
        self,
        *,
        blocks: list[GovernedBlock],
        choice_text: str,
        question: str,
    ) -> tuple[list[GovernedBlock], list[dict[str, Any]]]:
        if self.evidence_mode == "none" or not choice_text:
            return [], []

        exact = find_exact_choice_blocks(blocks, choice_text)
        if exact or self.evidence_mode == "choice_exact":
            return exact, [
                evidence_match(block, choice_text, 1.0, "longbench_choice_exact")
                for block in exact
            ]

        overlap = find_overlap_choice_blocks(blocks, choice_text, question)
        return [block for block, _score, _terms in overlap], [
            evidence_match(block, choice_text, score, "longbench_choice_overlap", matched_terms=terms)
            for block, score, terms in overlap
        ]


def normalize_answer_label(value: Any) -> str:
    text = normalize_text(value).upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else ""


def context_sections(context: str) -> tuple[list[tuple[list[str], str]], str]:
    stripped = context.strip()
    if not stripped:
        return [], "text"
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [([], context)], "text"
        sections = list(flatten_json_sections(parsed))
        if sections:
            return sections, "json"
    return [([], context)], "text"


def flatten_json_sections(value: Any, path: list[str] | None = None) -> Iterable[tuple[list[str], str]]:
    path = path or []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = normalize_text(key) or "field"
            yield from flatten_json_sections(item, [*path, key_text[:80]])
    elif isinstance(value, list):
        scalar_items = [item for item in value if not isinstance(item, (dict, list))]
        nested_items = [(index, item) for index, item in enumerate(value) if isinstance(item, (dict, list))]
        if scalar_items:
            text = "\n".join(normalize_text(item) for item in scalar_items if normalize_text(item))
            if text:
                yield path or ["list"], text
        for index, item in nested_items:
            yield from flatten_json_sections(item, [*path, f"item_{index}"])
    else:
        text = clean_context_text(value)
        if text:
            yield path or ["context"], text


def split_paragraphs(text: str) -> list[str]:
    text = clean_context_text(text)
    if not text:
        return []
    paragraphs = [item.strip() for item in PARAGRAPH_SPLIT_RE.split(text) if item.strip()]
    if len(paragraphs) == 1 and len(paragraphs[0]) > 4000:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return merge_lines_to_paragraphs(lines, max_chars=900)
    return paragraphs


def merge_lines_to_paragraphs(lines: list[str], max_chars: int) -> list[str]:
    paragraphs: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in lines:
        if is_heading(line) and buffer:
            paragraphs.append("\n".join(buffer))
            buffer = []
            size = 0
        if buffer and size + len(line) + 1 > max_chars:
            paragraphs.append("\n".join(buffer))
            buffer = []
            size = 0
        buffer.append(line)
        size += len(line) + 1
    if buffer:
        paragraphs.append("\n".join(buffer))
    return paragraphs


def split_long_block(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        return merge_lines_to_paragraphs(lines, max_chars=max_chars)

    sentences = [item.strip() for item in SENTENCE_SPLIT_RE.split(text) if item.strip()]
    if len(sentences) <= 1:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    pieces: list[str] = []
    buffer: list[str] = []
    size = 0
    for sentence in sentences:
        if buffer and size + len(sentence) + 1 > max_chars:
            pieces.append(" ".join(buffer))
            buffer = []
            size = 0
        buffer.append(sentence)
        size += len(sentence) + 1
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


def normalize_block_text(text: str, block_type: str) -> str:
    if block_type in {"code", "table"}:
        return text.strip()
    return normalize_text(text)


def infer_block_type(text: str) -> str:
    if looks_like_code(text):
        return "code"
    if looks_like_table(text):
        return "table"
    if looks_like_list(text):
        return "list"
    if is_heading(text):
        return "heading"
    return "paragraph"


def is_heading(text: str) -> bool:
    value = normalize_text(text)
    if not value or len(value) > 140:
        return False
    if "\n" in text and len([line for line in text.splitlines() if line.strip()]) > 2:
        return False
    if HEADING_RE.match(value):
        return True
    alpha = [char for char in value if char.isalpha()]
    return bool(alpha) and len(value.split()) <= 10 and sum(char.isupper() for char in alpha) / len(alpha) > 0.65


def looks_like_list(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    hits = sum(1 for line in lines if LIST_RE.match(line))
    return hits >= 2 and hits / len(lines) >= 0.45


def looks_like_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    delimited = sum(
        1
        for line in lines
        if line.count("|") >= 1 or line.count("\t") >= 1 or line.count(",") >= 4
    )
    return delimited >= 2 and delimited / len(lines) >= 0.35


def looks_like_code(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    code_hits = sum(
        1
        for line in lines
        if re.search(r"\b(def|class|import|from|return|if|else|for|while|function|const|let|var)\b", line)
        or line.strip().startswith(("{", "}", "</", "#include"))
        or line.count("{") + line.count("}") + line.count(";") >= 2
    )
    if code_hits >= 2 and code_hits / len(lines) >= 0.25:
        return True
    return "```" in text


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


def find_exact_choice_blocks(blocks: list[GovernedBlock], choice_text: str) -> list[GovernedBlock]:
    needle = normalize_text(choice_text).lower()
    if not needle:
        return []
    matches = [block for block in blocks if needle in normalize_text(block.text).lower()]
    return matches[:3]


def content_terms(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "into",
        "according",
        "through",
        "which",
        "what",
        "when",
        "where",
        "how",
        "why",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
    }
    return {term.lower() for term in WORD_RE.findall(text) if term.lower() not in stopwords}


def find_overlap_choice_blocks(
    blocks: list[GovernedBlock],
    choice_text: str,
    question: str,
) -> list[tuple[GovernedBlock, float, list[str]]]:
    answer_terms = content_terms(choice_text)
    if len(answer_terms) < 3:
        return []
    question_terms = content_terms(question)
    scored: list[tuple[float, GovernedBlock, list[str]]] = []
    for block in blocks:
        block_terms = content_terms(block.text)
        if not block_terms:
            continue
        matched_answer = sorted(answer_terms & block_terms)
        matched_question = sorted(question_terms & block_terms)
        answer_score = len(matched_answer) / len(answer_terms)
        question_score = len(matched_question) / max(1, len(question_terms))
        score = round((answer_score * 0.75) + (question_score * 0.25), 4)
        if len(matched_answer) >= 3 and score >= 0.42:
            scored.append((score, block, matched_answer[:20]))
    scored.sort(key=lambda item: (-item[0], item[1].order))
    return [(block, score, terms) for score, block, terms in scored[:3]]


def evidence_match(
    block: GovernedBlock,
    choice_text: str,
    score: float,
    method: str,
    *,
    matched_terms: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "evidence_text": block.text,
        "block_id": block.block_id,
        "score": score,
        "method": method,
        "weak_evidence": True,
        "choice_text": choice_text,
        "matched_terms": matched_terms or [],
        "section_name": block.source_anchor.section_name,
        "paragraph_index": block.source_anchor.paragraph_index,
    }


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
