# -*- coding: utf-8 -*-
"""Validate generated RAG chunks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", help="Path to chunks_*.jsonl")
    parser.add_argument(
        "--blocks",
        default=None,
        help="Optional blocks.jsonl path for source block existence validation.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_path = Path(args.chunks)
    chunks = read_jsonl(chunk_path)
    block_ids = None
    if args.blocks:
        block_ids = {row["block_id"] for row in read_jsonl(Path(args.blocks))}

    errors: list[str] = []
    warnings: list[str] = []
    flag_counts: Counter[str] = Counter()
    token_counts: list[int] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        token_count = chunk.get("token_count") or 0
        token_counts.append(token_count)
        flag_counts.update(chunk.get("quality_flags") or [])

        if not chunk.get("text"):
            errors.append(f"{chunk_id}: empty text")
        if not chunk.get("source_blocks"):
            errors.append(f"{chunk_id}: missing source_blocks")
        if not chunk.get("source_anchor"):
            errors.append(f"{chunk_id}: missing source_anchor")
        if not chunk.get("quality_flags"):
            warnings.append(f"{chunk_id}: missing quality_flags")
        if args.max_tokens and token_count > args.max_tokens:
            warnings.append(f"{chunk_id}: token_count {token_count} > max_tokens {args.max_tokens}")
        if block_ids is not None:
            for block_id in chunk.get("source_blocks") or []:
                if block_id not in block_ids:
                    errors.append(f"{chunk_id}: unknown source block {block_id}")

    report = {
        "chunks_path": str(chunk_path),
        "chunks": len(chunks),
        "min_tokens": min(token_counts) if token_counts else None,
        "max_tokens": max(token_counts) if token_counts else None,
        "avg_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else None,
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "errors": errors,
        "warnings": warnings[:100],
        "status": "passed" if not errors else "failed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

