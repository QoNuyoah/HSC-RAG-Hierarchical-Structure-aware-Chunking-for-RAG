# -*- coding: utf-8 -*-
"""Convert local LongBench-v2 CSV-in-zip artifacts to HSC-RAG GovernedDocument artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.adapters.longbench_adapter import LongBenchAdapter, write_jsonl, write_queries_csv  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "LongBench.zip"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to LongBench.zip.")
    parser.add_argument("--member", default=None, help="CSV member inside the zip. Defaults to longbench-v2.csv.")
    parser.add_argument("--split", default="eval", help="Split label to write into GovernedDocument.")
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=20,
        help="Number of LongBench rows to convert. Use 0 or negative for all rows.",
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=1800,
        help="Maximum approximate characters per generated GovernedBlock.",
    )
    parser.add_argument(
        "--max-blocks-per-doc",
        type=int,
        default=0,
        help="Optional cap for generated blocks per LongBench document. Use 0 for no cap.",
    )
    parser.add_argument(
        "--evidence-mode",
        default="choice_overlap",
        choices=["none", "choice_exact", "choice_overlap"],
        help=(
            "How to derive weak retrieval evidence. LongBench-v2 has answer labels "
            "but no annotated evidence; choice_overlap is weak and reported as such."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/processed/longbench_v2/{split}_{limit}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    limit_docs = args.limit_docs if args.limit_docs and args.limit_docs > 0 else None
    limit_label = str(limit_docs) if limit_docs is not None else "all"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "processed" / "longbench_v2" / f"{args.split}_{limit_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = LongBenchAdapter(
        input_path,
        member=args.member,
        max_block_chars=args.max_block_chars,
        max_blocks_per_doc=args.max_blocks_per_doc,
        evidence_mode=args.evidence_mode,
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
    report["member"] = adapter.member
    report["output_dir"] = display_path(output_dir)
    report["limit_docs"] = limit_docs
    report["max_block_chars"] = args.max_block_chars
    report["max_blocks_per_doc"] = adapter.max_blocks_per_doc
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
