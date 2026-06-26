# -*- coding: utf-8 -*-
"""Convert local HotpotQA.zip to HSC-RAG GovernedDocument artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.adapters.hotpotqa_adapter import HotpotQAAdapter, write_jsonl, write_queries_csv  # noqa: E402


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        default=str(PROJECT_ROOT / "HotpotQA.zip"),
        help="Path to local HotpotQA.zip.",
    )
    parser.add_argument(
        "--member",
        default=None,
        help="JSON member inside the zip. Defaults to the first .json member.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=50,
        help="Number of QA records to convert. Keep small for fast experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to data/processed/hotpotqa/{split}_{limit}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "processed" / "hotpotqa" / f"{args.split}_{args.limit_docs}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = HotpotQAAdapter(args.zip, member=args.member)
    docs, evidence_records, stats = adapter.convert(
        split=args.split,
        limit_docs=args.limit_docs,
    )

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
    report["zip_path"] = display_path(Path(args.zip))
    report["member"] = adapter.member
    report["output_dir"] = display_path(output_dir)
    report["limit_docs"] = args.limit_docs
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
