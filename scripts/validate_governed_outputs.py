# -*- coding: utf-8 -*-
"""Validate HSC-RAG GovernedDocument conversion artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = [
    "governed_documents.jsonl",
    "blocks.jsonl",
    "queries.csv",
    "gold_evidence.jsonl",
    "conversion_report.json",
]


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=r"E:\practical_training\HSC_RAG\data\processed\qasper\train",
        help="Directory containing conversion artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files in {input_dir}: {missing}")

    docs = read_jsonl(input_dir / "governed_documents.jsonl")
    blocks = read_jsonl(input_dir / "blocks.jsonl")
    evidence = read_jsonl(input_dir / "gold_evidence.jsonl")

    with (input_dir / "queries.csv").open("r", encoding="utf-8-sig", newline="") as f:
        queries = list(csv.DictReader(f))

    block_ids = {b["block_id"] for b in blocks}
    doc_ids = {d["doc_id"] for d in docs}

    errors: list[str] = []
    warnings: list[str] = []

    for doc in docs:
        if doc.get("normalization_status") not in {
            "provided_by_dataset",
            "provided_by_upstream",
            "simulated_governed",
        }:
            errors.append(f"{doc.get('doc_id')}: invalid normalization_status")
        if doc.get("governance_stage") != "post_normalization_packaging":
            errors.append(f"{doc.get('doc_id')}: invalid governance_stage")
        if not doc.get("blocks"):
            errors.append(f"{doc.get('doc_id')}: empty blocks")

    for block in blocks:
        if block.get("doc_id") not in doc_ids:
            errors.append(f"{block.get('block_id')}: doc_id not found")
        if not block.get("source_anchor"):
            errors.append(f"{block.get('block_id')}: missing source_anchor")
        if block.get("type") not in {"title", "heading"} and not block.get("title_path"):
            warnings.append(f"{block.get('block_id')}: missing title_path")

    evidence_items = 0
    matched_evidence_items = 0
    for record in evidence:
        if record.get("doc_id") not in doc_ids:
            errors.append(f"{record.get('query_id')}: doc_id not found")
        gold_block_ids = record.get("gold_block_ids") or []
        for block_id in gold_block_ids:
            if block_id not in block_ids:
                errors.append(f"{record.get('query_id')}: unknown gold block {block_id}")
        for match in record.get("evidence_matches") or []:
            evidence_items += 1
            if match.get("block_id"):
                matched_evidence_items += 1

    report = {
        "input_dir": str(input_dir),
        "documents": len(docs),
        "blocks": len(blocks),
        "queries": len(queries),
        "gold_evidence_records": len(evidence),
        "evidence_items": evidence_items,
        "matched_evidence_items": matched_evidence_items,
        "evidence_match_rate": (
            matched_evidence_items / evidence_items if evidence_items else None
        ),
        "errors": errors,
        "warnings": warnings[:50],
        "status": "passed" if not errors else "failed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

