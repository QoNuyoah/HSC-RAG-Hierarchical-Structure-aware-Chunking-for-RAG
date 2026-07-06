# -*- coding: utf-8 -*-
"""Convert local CJRC/CAIL2019 reading-comprehension JSON to HSC-RAG artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.adapters.cjrc_adapter import CjrcAdapter, write_jsonl, write_queries_csv  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cjrc_cail2019_reading_comprehension"
SPLIT_FILES = {
    "train": "big_train_data.json",
    "dev": "dev_ground_truth.json",
    "test": "test_ground_truth.json",
}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def default_input_path(split: str) -> Path:
    return RAW_DIR / SPLIT_FILES.get(split, f"{split}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Path to CJRC/CAIL2019 reading-comprehension JSON.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=50,
        help="Number of legal cases to convert. Use 0 or negative for all cases.",
    )
    parser.add_argument(
        "--block-mode",
        default="sentence",
        choices=["sentence", "paragraph"],
        help="How to turn each legal context into GovernedBlock units.",
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=260,
        help="Maximum approximate characters per generated content block.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/processed/cjrc/{split}_{limit}_{block_mode}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else default_input_path(args.split)
    limit_docs = args.limit_docs if args.limit_docs and args.limit_docs > 0 else None
    limit_label = str(limit_docs) if limit_docs is not None else "all"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "processed" / "cjrc" / f"{args.split}_{limit_label}_{args.block_mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = CjrcAdapter(
        input_path,
        block_mode=args.block_mode,
        max_block_chars=args.max_block_chars,
    )
    docs, evidence_records, stats = adapter.convert(split=args.split, limit_docs=limit_docs)

    write_jsonl(output_dir / "governed_documents.jsonl", (doc.model_dump(mode="json") for doc in docs))
    write_jsonl(
        output_dir / "blocks.jsonl",
        (block.model_dump(mode="json") for doc in docs for block in doc.blocks),
    )
    write_jsonl(
        output_dir / "gold_evidence.jsonl",
        (record.model_dump(mode="json") for record in evidence_records),
    )
    write_queries_csv(output_dir / "queries.csv", (query for doc in docs for query in doc.queries))

    report = stats.to_dict()
    report["input_path"] = display_path(input_path)
    report["output_dir"] = display_path(output_dir)
    report["limit_docs"] = limit_docs
    report["block_mode"] = args.block_mode
    report["max_block_chars"] = args.max_block_chars
    report["artifacts"] = {
        "governed_documents": "governed_documents.jsonl",
        "blocks": "blocks.jsonl",
        "queries": "queries.csv",
        "gold_evidence": "gold_evidence.jsonl",
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
