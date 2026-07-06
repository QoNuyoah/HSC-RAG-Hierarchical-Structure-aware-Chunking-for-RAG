from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from app.adapters.longbench_adapter import LongBenchAdapter


def _write_longbench_zip(path: Path, rows: list[dict[str, str]]) -> None:
    csv_path = path.with_suffix(".csv")
    fieldnames = [
        "_id",
        "domain",
        "sub_domain",
        "difficulty",
        "length",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer",
        "context",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(csv_path, "longbench-v2.csv")


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "_id": "sample_001",
        "domain": "Single-Document QA",
        "sub_domain": "Governmental",
        "difficulty": "easy",
        "length": "short",
        "question": "How should the smart court construction be promoted?",
        "choice_A": "By changing office decorations.",
        "choice_B": "By reducing public services.",
        "choice_C": "By replacing all courts.",
        "choice_D": "Use advanced information systems to improve the level of information technology in case handling.",
        "answer": "D",
        "context": (
            "Contents\n\n"
            "I. Promoting smart courts\n\n"
            "The report says that courts should use advanced information systems "
            "to improve the level of information technology in case handling."
        ),
    }
    row.update(overrides)
    return row


def test_longbench_adapter_maps_csv_context_to_governed_document_and_weak_evidence(tmp_path):
    zip_path = tmp_path / "longbench.zip"
    _write_longbench_zip(zip_path, [_row()])

    docs, evidence, stats = LongBenchAdapter(
        zip_path,
        evidence_mode="choice_exact",
        max_block_chars=500,
    ).convert(split="eval", limit_docs=1)

    assert stats.documents == 1
    assert stats.blocks >= 2
    assert stats.weak_evidence_queries == 1
    assert len(docs) == 1
    doc = docs[0]
    assert doc.dataset == "longbench_v2"
    assert doc.queries[0].answer.startswith("D:")
    assert doc.queries[0].gold_block_ids
    assert doc.queries[0].metadata["evidence_is_weak"] is True
    assert evidence[0].evidence_matches[0]["method"] == "longbench_choice_exact"
    assert evidence[0].evidence_matches[0]["weak_evidence"] is True
    assert evidence[0].gold_block_ids == doc.queries[0].gold_block_ids


def test_longbench_adapter_does_not_fabricate_evidence_when_choice_is_absent(tmp_path):
    zip_path = tmp_path / "longbench.zip"
    _write_longbench_zip(
        zip_path,
        [
            _row(
                context="This document discusses unrelated administrative process details.",
            )
        ],
    )

    docs, evidence, stats = LongBenchAdapter(
        zip_path,
        evidence_mode="choice_exact",
    ).convert(split="eval", limit_docs=1)

    query = docs[0].queries[0]
    assert stats.weak_evidence_queries == 0
    assert query.is_unanswerable is True
    assert query.metadata["longbench_answerable"] is True
    assert query.metadata["retrieval_evaluable"] is False
    assert evidence[0].gold_block_ids == []
    assert evidence[0].evidence_matches == []


def test_longbench_adapter_preserves_json_context_structure_in_title_path(tmp_path):
    zip_path = tmp_path / "longbench.zip"
    context = json.dumps(
        {
            "manual": {
                "overview": "Chapter One\n\nThe alpha system contains a retriever and a chunker.",
                "table": "name|value\nretriever|bm25\nchunker|hsc-rag",
            }
        }
    )
    _write_longbench_zip(
        zip_path,
        [
            _row(
                _id="json_sample",
                domain="Long Structured Data Understanding",
                sub_domain="Table QA",
                question="Which component uses bm25?",
                choice_D="retriever",
                answer="D",
                context=context,
            )
        ],
    )

    docs, _evidence, stats = LongBenchAdapter(
        zip_path,
        evidence_mode="choice_overlap",
        max_block_chars=500,
    ).convert(split="eval", limit_docs=1)

    doc = docs[0]
    assert stats.json_contexts == 1
    assert doc.metadata["context_format"] == "json"
    assert any("manual" in block.title_path for block in doc.blocks)
    assert any(block.type == "table" for block in doc.blocks)
