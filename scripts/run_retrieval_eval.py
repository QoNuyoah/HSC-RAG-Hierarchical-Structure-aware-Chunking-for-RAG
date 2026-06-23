# -*- coding: utf-8 -*-
"""Evaluate retrieval over generated chunk files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.retrievers.bm25 import BM25ChunkRetriever  # noqa: E402
from app.retrievers.dense_faiss import DenseFaissRetriever  # noqa: E402
from app.retrievers.hybrid import HybridRetriever  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "qasper" / "train"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing chunks_{strategy}.jsonl.",
    )
    parser.add_argument(
        "--gold-evidence",
        default=str(DEFAULT_DATA_DIR / "gold_evidence.jsonl"),
        help="Path to gold_evidence.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to --chunk-dir.",
    )
    parser.add_argument(
        "--strategies",
        default="fixed,recursive,semantic,hsc_rag",
        help="Comma-separated chunk strategies to evaluate.",
    )
    parser.add_argument(
        "--retrievers",
        default="bm25,dense,hybrid",
        help="Comma-separated retrievers: bm25,dense,hybrid.",
    )
    parser.add_argument(
        "--top-k",
        default="1,3,5",
        help="Comma-separated Recall/Hit cutoffs.",
    )
    parser.add_argument("--ndcg-k", type=int, default=5)
    parser.add_argument(
        "--global-search",
        action="store_true",
        help="Search all chunks instead of filtering candidates to the query document.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Index chunk title_path/tags/summary in addition to chunk text.",
    )
    parser.add_argument(
        "--dense-encoder",
        default="tfidf_svd",
        choices=["tfidf_svd", "sentence_transformer", "auto"],
        help="Dense encoder. tfidf_svd is local and deterministic; auto falls back if model is unavailable.",
    )
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name when using dense-encoder sentence_transformer/auto.",
    )
    parser.add_argument("--dense-svd-dim", type=int, default=128)
    parser.add_argument("--hybrid-alpha", type=float, default=0.55, help="BM25 weight in hybrid fusion.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow SentenceTransformer to look beyond the local HuggingFace cache.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_top_k(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top-k must contain positive integers")
    return values


def gold_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = read_jsonl(path)
    skipped = {"unanswerable": 0, "no_gold_blocks": 0}
    answerable: list[dict[str, Any]] = []
    for record in records:
        if record.get("is_unanswerable"):
            skipped["unanswerable"] += 1
            continue
        if not record.get("gold_block_ids"):
            skipped["no_gold_blocks"] += 1
            continue
        answerable.append(record)
    return answerable, skipped


def relevance(hit: dict[str, Any] | Any, gold_blocks: set[str]) -> int:
    source_blocks = getattr(hit, "source_blocks", None)
    if source_blocks is None:
        source_blocks = hit.get("source_blocks") or []
    return 1 if gold_blocks.intersection(source_blocks) else 0


def covered_blocks(hits: list[Any], gold_blocks: set[str], k: int) -> set[str]:
    covered: set[str] = set()
    for hit in hits[:k]:
        covered.update(gold_blocks.intersection(hit.source_blocks))
    return covered


def dcg(relevances: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def ndcg_at_k(
    *,
    hits: list[Any],
    candidate_chunks: list[dict[str, Any]],
    gold_blocks: set[str],
    k: int,
) -> float:
    ranked_rels = [relevance(hit, gold_blocks) for hit in hits[:k]]
    ideal_rels = sorted((relevance(chunk, gold_blocks) for chunk in candidate_chunks), reverse=True)[:k]
    ideal = dcg(ideal_rels, k)
    if ideal <= 0:
        return 0.0
    return dcg(ranked_rels, k) / ideal


def build_retriever(
    *,
    retriever_name: str,
    chunks: list[dict[str, Any]],
    include_metadata: bool,
    dense_encoder: str,
    dense_model: str,
    dense_svd_dim: int,
    hybrid_alpha: float,
    local_files_only: bool,
):
    if retriever_name == "bm25":
        return BM25ChunkRetriever(chunks, include_metadata=include_metadata)
    if retriever_name == "dense":
        return DenseFaissRetriever(
            chunks,
            encoder=dense_encoder,
            model_name=dense_model,
            include_metadata=include_metadata,
            svd_dim=dense_svd_dim,
            local_files_only=local_files_only,
        )
    if retriever_name == "hybrid":
        return HybridRetriever(
            chunks,
            alpha=hybrid_alpha,
            include_metadata=include_metadata,
            dense_encoder=dense_encoder,
            dense_model_name=dense_model,
            dense_svd_dim=dense_svd_dim,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unknown retriever: {retriever_name}")


def retriever_config(retriever_name: str, retriever: Any) -> dict[str, Any]:
    base = {"retriever": retriever_name}
    if hasattr(retriever, "config"):
        base.update(retriever.config())
    return base


def evaluate_pair(
    *,
    strategy: str,
    retriever_name: str,
    chunks_path: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    skipped_counts: dict[str, int],
    top_ks: list[int],
    ndcg_k: int,
    doc_filter: bool,
    retriever: Any,
    include_metadata: bool,
) -> dict[str, Any]:
    chunks = read_jsonl(chunks_path)
    max_k = max(max(top_ks), ndcg_k)

    per_query: list[dict[str, Any]] = []
    aggregate_recalls = {k: [] for k in top_ks}
    aggregate_hits = {k: [] for k in top_ks}
    aggregate_full_recall = {k: [] for k in top_ks}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for record in records:
        gold_blocks = set(record["gold_block_ids"])
        query_doc_id = record.get("doc_id") if doc_filter else None
        candidate_chunks = retriever.candidate_chunks(query_doc_id)
        hits = retriever.search(
            record["question"],
            top_k=len(candidate_chunks),
            doc_id=query_doc_id,
        )

        first_relevant_rank = 0
        for hit in hits:
            if gold_blocks.intersection(hit.source_blocks):
                first_relevant_rank = hit.rank
                break
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)
        query_ndcg = ndcg_at_k(
            hits=hits,
            candidate_chunks=candidate_chunks,
            gold_blocks=gold_blocks,
            k=ndcg_k,
        )
        ndcgs.append(query_ndcg)

        recall_by_k: dict[str, float] = {}
        hit_by_k: dict[str, int] = {}
        full_recall_by_k: dict[str, int] = {}
        covered_by_k: dict[str, list[str]] = {}
        for k in top_ks:
            covered = covered_blocks(hits, gold_blocks, k)
            recall = len(covered) / len(gold_blocks) if gold_blocks else 0.0
            hit = 1 if covered else 0
            full_recall = 1 if covered == gold_blocks else 0
            recall_by_k[str(k)] = recall
            hit_by_k[str(k)] = hit
            full_recall_by_k[str(k)] = full_recall
            covered_by_k[str(k)] = sorted(covered)
            aggregate_recalls[k].append(recall)
            aggregate_hits[k].append(hit)
            aggregate_full_recall[k].append(full_recall)

        per_query.append(
            {
                "query_id": record["query_id"],
                "doc_id": record["doc_id"],
                "question": record["question"],
                "gold_block_ids": sorted(gold_blocks),
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": reciprocal_ranks[-1],
                f"ndcg@{ndcg_k}": query_ndcg,
                "recall_by_k": recall_by_k,
                "hit_by_k": hit_by_k,
                "full_recall_by_k": full_recall_by_k,
                "covered_gold_blocks_by_k": covered_by_k,
                "missing_gold_blocks_at_max_k": sorted(gold_blocks - covered_blocks(hits, gold_blocks, max(top_ks))),
                "top_hits": [
                    {
                        **hit.to_dict(),
                        "is_relevant": bool(gold_blocks.intersection(hit.source_blocks)),
                        "covered_gold_block_ids": sorted(gold_blocks.intersection(hit.source_blocks)),
                    }
                    for hit in hits[:max_k]
                ],
            }
        )

    result_path = output_dir / f"retrieval_results_{strategy}_{retriever_name}.jsonl"
    write_jsonl(result_path, per_query)

    def mean(values: list[float | int]) -> float:
        return round(sum(values) / len(values), 6) if values else 0.0

    metrics: dict[str, Any] = {
        "strategy": strategy,
        "retriever": retriever_name,
        "chunks_path": str(chunks_path),
        "results_path": str(result_path),
        "chunks": len(chunks),
        "queries_total_in_gold_file": len(records) + sum(skipped_counts.values()),
        "queries_evaluated": len(records),
        "queries_skipped": skipped_counts,
        "search_scope": "same_doc" if doc_filter else "global",
        "index_fields": ["text", "title_path", "tags", "summary"] if include_metadata else ["text"],
        "retriever_config": retriever_config(retriever_name, retriever),
        "metric_definitions": {
            "recall@k": "Mean block-level evidence recall: covered gold_block_ids in top-k chunks divided by all gold_block_ids.",
            "hit_rate@k": "Mean query-level hit rate: at least one gold evidence block appears in top-k chunks.",
            "full_recall_rate@k": "Mean query-level complete evidence coverage: all gold evidence blocks appear in top-k chunks.",
            "mrr": "Mean reciprocal rank of the first chunk containing any gold evidence block.",
            f"ndcg@{ndcg_k}": "Binary chunk-relevance nDCG: a chunk is relevant if it contains at least one gold evidence block.",
        },
        "metrics": {},
    }

    for k in top_ks:
        metrics["metrics"][f"recall@{k}"] = mean(aggregate_recalls[k])
        metrics["metrics"][f"hit_rate@{k}"] = mean(aggregate_hits[k])
        metrics["metrics"][f"full_recall_rate@{k}"] = mean(aggregate_full_recall[k])
    metrics["metrics"]["mrr"] = mean(reciprocal_ranks)
    metrics["metrics"][f"ndcg@{ndcg_k}"] = mean(ndcgs)
    metrics["metrics"]["avg_gold_blocks_per_query"] = mean([len(record["gold_block_ids"]) for record in records])

    metrics_path = output_dir / f"retrieval_eval_{strategy}_{retriever_name}.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def compare_against_fixed(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for report in reports:
        grouped.setdefault(report["retriever"], {})[report["strategy"]] = report

    comparisons: list[dict[str, Any]] = []
    for retriever_name, by_strategy in grouped.items():
        baseline = by_strategy.get("fixed")
        if baseline is None:
            continue
        for target_name, target in sorted(by_strategy.items()):
            if target_name == "fixed":
                continue
            deltas = {
                key: round(target["metrics"][key] - baseline["metrics"][key], 6)
                for key in baseline["metrics"]
                if key in target["metrics"] and isinstance(baseline["metrics"][key], (int, float))
            }
            comparisons.append(
                {
                    "retriever": retriever_name,
                    "baseline": "fixed",
                    "target": target_name,
                    "delta_target_minus_baseline": deltas,
                }
            )
    return comparisons


def best_by_metric(reports: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    keys = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
    result: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        ranked = sorted(
            (
                {
                    "strategy": report["strategy"],
                    "retriever": report["retriever"],
                    "value": report["metrics"].get(key, 0.0),
                }
                for report in reports
            ),
            key=lambda item: item["value"],
            reverse=True,
        )
        result[key] = ranked[:5]
    return result


def main() -> None:
    args = parse_args()
    chunk_dir = Path(args.chunk_dir)
    output_dir = Path(args.output_dir) if args.output_dir else chunk_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = parse_top_k(args.top_k)
    records, skipped = gold_records(Path(args.gold_evidence))
    strategies = parse_csv_list(args.strategies)
    retriever_names = parse_csv_list(args.retrievers)
    doc_filter = not args.global_search
    local_files_only = not args.allow_model_download

    reports: list[dict[str, Any]] = []
    for strategy in strategies:
        chunks_path = chunk_dir / f"chunks_{strategy}.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunk file: {chunks_path}")
        chunks = read_jsonl(chunks_path)
        for retriever_name in retriever_names:
            retriever = build_retriever(
                retriever_name=retriever_name,
                chunks=chunks,
                include_metadata=args.include_metadata,
                dense_encoder=args.dense_encoder,
                dense_model=args.dense_model,
                dense_svd_dim=args.dense_svd_dim,
                hybrid_alpha=args.hybrid_alpha,
                local_files_only=local_files_only,
            )
            reports.append(
                evaluate_pair(
                    strategy=strategy,
                    retriever_name=retriever_name,
                    chunks_path=chunks_path,
                    output_dir=output_dir,
                    records=records,
                    skipped_counts=skipped,
                    top_ks=top_ks,
                    ndcg_k=args.ndcg_k,
                    doc_filter=doc_filter,
                    retriever=retriever,
                    include_metadata=args.include_metadata,
                )
            )

    summary = {
        "gold_evidence_path": str(Path(args.gold_evidence)),
        "output_dir": str(output_dir),
        "strategies": strategies,
        "retrievers": retriever_names,
        "reports": reports,
        "comparisons_against_fixed": compare_against_fixed(reports),
        "best_by_metric": best_by_metric(reports),
    }
    if retriever_names == ["bm25"]:
        summary_path = output_dir / "retrieval_eval_bm25_summary.json"
    else:
        summary_path = output_dir / "retrieval_eval_multi_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

