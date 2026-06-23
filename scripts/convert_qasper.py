# -*- coding: utf-8 -*-
"""Convert local QASPER.zip to HSC-RAG GovernedDocument artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.adapters.qasper_adapter import QasperAdapter, write_jsonl, write_queries_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        default=str(PROJECT_ROOT / "QASPER.zip"),
        help="Path to local QASPER.zip. Defaults to project_root/QASPER.zip.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="QASPER split to convert.",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=5,
        help="Limit documents for quick local runs. Use 0 or negative for all docs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "qasper"),
        help="Output directory for converted artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit_docs = args.limit_docs if args.limit_docs and args.limit_docs > 0 else None
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = QasperAdapter(args.zip)
    documents, evidence_records, stats = adapter.iter_documents(
        split=args.split,
        limit_docs=limit_docs,
    )

    all_blocks = [block for doc in documents for block in doc.blocks]
    all_queries = [query for doc in documents for query in doc.queries]

    write_jsonl(
        output_dir / "governed_documents.jsonl",
        (doc.model_dump(mode="json") for doc in documents),
    )
    write_jsonl(
        output_dir / "blocks.jsonl",
        (block.model_dump(mode="json") for block in all_blocks),
    )
    write_queries_csv(output_dir / "queries.csv", all_queries)
    write_jsonl(
        output_dir / "gold_evidence.jsonl",
        (record.model_dump(mode="json") for record in evidence_records),
    )

    report = stats.to_dict()
    report.update(
        {
            "zip_path": str(Path(args.zip).resolve()),
            "output_dir": str(output_dir.resolve()),
            "limit_docs": limit_docs,
            "artifacts": {
                "governed_documents": "governed_documents.jsonl",
                "blocks": "blocks.jsonl",
                "queries": "queries.csv",
                "gold_evidence": "gold_evidence.jsonl",
                "conversion_report": "conversion_report.json",
            },
        }
    )
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

