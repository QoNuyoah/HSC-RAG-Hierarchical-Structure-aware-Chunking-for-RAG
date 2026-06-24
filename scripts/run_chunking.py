# -*- coding: utf-8 -*-
"""Run chunking strategies over GovernedDocument artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.chunkers.common import load_governed_documents, write_jsonl  # noqa: E402
from app.chunkers.fixed import FixedChunkConfig, FixedSizeChunker  # noqa: E402
from app.chunkers.hsc_rag import HscRagChunker, HscRagConfig  # noqa: E402
from app.chunkers.recursive import RecursiveChunkConfig, RecursiveChunker  # noqa: E402
from app.chunkers.semantic import SemanticChunkConfig, SemanticChunker  # noqa: E402
from app.core.schemas import ChunkRunReport, RagChunk  # noqa: E402


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "qasper" / "train" / "governed_documents.jsonl"),
        help="Path to governed_documents.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to input file directory.",
    )
    parser.add_argument(
        "--strategies",
        default="fixed,recursive,semantic,hsc_rag",
        help="Comma-separated strategies: fixed,recursive,semantic,hsc_rag.",
    )
    parser.add_argument("--fixed-target", type=int, default=512)
    parser.add_argument("--fixed-overlap", type=int, default=64)
    parser.add_argument("--recursive-target", type=int, default=512)
    parser.add_argument("--recursive-overlap", type=int, default=64)
    parser.add_argument("--semantic-min", type=int, default=160)
    parser.add_argument("--semantic-target", type=int, default=512)
    parser.add_argument("--semantic-max", type=int, default=768)
    parser.add_argument("--semantic-breakpoint-percentile", type=float, default=75.0)
    parser.add_argument("--hsc-min", type=int, default=180)
    parser.add_argument("--hsc-target", type=int, default=512)
    parser.add_argument("--hsc-max", type=int, default=900)
    return parser.parse_args()


def report_for(
    *,
    strategy: str,
    input_path: Path,
    output_path: Path,
    documents: int,
    chunks: list[RagChunk],
    config: dict,
) -> ChunkRunReport:
    token_counts = [chunk.token_count for chunk in chunks]
    flag_counts: Counter[str] = Counter()
    for chunk in chunks:
        flag_counts.update(chunk.quality_flags)
    total_tokens = sum(token_counts)
    return ChunkRunReport(
        strategy=strategy,  # type: ignore[arg-type]
        input_path=display_path(input_path),
        output_path=display_path(output_path),
        documents=documents,
        chunks=len(chunks),
        total_tokens=total_tokens,
        avg_tokens=round(total_tokens / len(chunks), 2) if chunks else None,
        min_tokens=min(token_counts) if token_counts else None,
        max_tokens=max(token_counts) if token_counts else None,
        quality_flag_counts=dict(sorted(flag_counts.items())),
        config=config,
    )


def run_strategy(strategy: str, docs, args: argparse.Namespace) -> tuple[list[RagChunk], dict]:
    if strategy == "fixed":
        config = FixedChunkConfig(
            target_tokens=args.fixed_target,
            max_tokens=args.fixed_target,
            overlap_tokens=args.fixed_overlap,
        )
        chunker = FixedSizeChunker(config)
        config_dict = config.__dict__
    elif strategy == "recursive":
        config = RecursiveChunkConfig(
            target_tokens=args.recursive_target,
            max_tokens=args.recursive_target,
            overlap_tokens=args.recursive_overlap,
        )
        chunker = RecursiveChunker(config)
        config_dict = config.__dict__
    elif strategy == "semantic":
        config = SemanticChunkConfig(
            min_tokens=args.semantic_min,
            target_tokens=args.semantic_target,
            max_tokens=args.semantic_max,
            breakpoint_percentile=args.semantic_breakpoint_percentile,
        )
        chunker = SemanticChunker(config)
        config_dict = config.__dict__
    elif strategy == "hsc_rag":
        config = HscRagConfig(
            min_tokens=args.hsc_min,
            target_tokens=args.hsc_target,
            max_tokens=args.hsc_max,
        )
        chunker = HscRagChunker(config)
        config_dict = config.__dict__
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    chunks: list[RagChunk] = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))
    return chunks, dict(config_dict)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = load_governed_documents(input_path)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    all_reports = []
    for strategy in strategies:
        chunks, config = run_strategy(strategy, docs, args)
        chunk_path = output_dir / f"chunks_{strategy}.jsonl"
        report_path = output_dir / f"chunk_report_{strategy}.json"
        write_jsonl(chunk_path, (chunk.model_dump(mode="json") for chunk in chunks))
        report = report_for(
            strategy=strategy,
            input_path=input_path,
            output_path=chunk_path,
            documents=len(docs),
            chunks=chunks,
            config=config,
        )
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_reports.append(report.model_dump(mode="json"))

    summary_path = output_dir / "chunking_summary.json"
    summary_path.write_text(
        json.dumps({"reports": all_reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"reports": all_reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
