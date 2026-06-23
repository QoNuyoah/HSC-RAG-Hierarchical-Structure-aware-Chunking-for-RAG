# -*- coding: utf-8 -*-
"""Read chunking and retrieval evaluation artifacts for the API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "qasper" / "train"
STRATEGIES = ["fixed", "recursive", "semantic", "hsc_rag"]
RETRIEVERS = ["bm25", "dense", "hybrid"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EvaluationStore:
    """File-backed read model for HSC-RAG evaluation artifacts."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = data_dir

    @lru_cache(maxsize=16)
    def overview(self) -> dict[str, Any]:
        conversion = _read_json(self.data_dir / "conversion_report.json")
        chunking = _read_json(self.data_dir / "chunking_summary.json")
        retrieval = _read_json(self.data_dir / "retrieval_eval_multi_summary.json")
        chunk_reports = {
            report["strategy"]: report
            for report in chunking.get("reports", [])
            if "strategy" in report
        }
        retrieval_reports = retrieval.get("reports", [])
        return {
            "project": {
                "name": "HSC-RAG",
                "title": "Hierarchical Structure-aware Chunking for RAG",
                "governance_stage": "post_normalization_packaging",
                "dataset": "QASPER",
                "split": "train",
            },
            "data_dir": str(self.data_dir),
            "conversion": conversion,
            "strategies": STRATEGIES,
            "retrievers": RETRIEVERS,
            "chunk_reports": chunk_reports,
            "retrieval_reports": retrieval_reports,
            "best_by_metric": retrieval.get("best_by_metric", {}),
            "comparisons_against_fixed": retrieval.get("comparisons_against_fixed", []),
        }

    @lru_cache(maxsize=32)
    def metrics(self, retriever: str | None = None) -> dict[str, Any]:
        reports = self.overview().get("retrieval_reports", [])
        if retriever:
            reports = [report for report in reports if report.get("retriever") == retriever]
        rows = []
        for report in reports:
            metrics = report.get("metrics", {})
            rows.append(
                {
                    "strategy": report.get("strategy"),
                    "retriever": report.get("retriever"),
                    "chunks": report.get("chunks"),
                    "queries_evaluated": report.get("queries_evaluated"),
                    "search_scope": report.get("search_scope"),
                    "index_fields": report.get("index_fields", []),
                    "recall@1": metrics.get("recall@1", 0.0),
                    "recall@3": metrics.get("recall@3", 0.0),
                    "recall@5": metrics.get("recall@5", 0.0),
                    "mrr": metrics.get("mrr", 0.0),
                    "ndcg@5": metrics.get("ndcg@5", 0.0),
                    "hit_rate@5": metrics.get("hit_rate@5", 0.0),
                    "full_recall_rate@5": metrics.get("full_recall_rate@5", 0.0),
                }
            )
        return {"retriever": retriever, "rows": rows}

    @lru_cache(maxsize=32)
    def query_index(self, retriever: str) -> dict[str, dict[str, dict[str, Any]]]:
        index: dict[str, dict[str, dict[str, Any]]] = {}
        for strategy in STRATEGIES:
            path = self.data_dir / f"retrieval_results_{strategy}_{retriever}.jsonl"
            for row in _read_jsonl(path):
                query_id = row.get("query_id")
                if not query_id:
                    continue
                index.setdefault(query_id, {})[strategy] = row
        return index

    @lru_cache(maxsize=64)
    def queries(self, retriever: str = "bm25") -> dict[str, Any]:
        index = self.query_index(retriever)
        items = []
        for query_id, by_strategy in sorted(index.items()):
            base = self._base_query_record(by_strategy)
            strategies = {}
            for strategy in STRATEGIES:
                row = by_strategy.get(strategy)
                if not row:
                    continue
                recall5 = (row.get("recall_by_k") or {}).get("5", 0.0)
                hit5 = (row.get("hit_by_k") or {}).get("5", 0)
                missing = row.get("missing_gold_blocks_at_max_k") or []
                strategies[strategy] = {
                    "first_relevant_rank": row.get("first_relevant_rank", 0),
                    "reciprocal_rank": row.get("reciprocal_rank", 0.0),
                    "recall@5": recall5,
                    "hit@5": hit5,
                    "missing_gold_count": len(missing),
                    "relevant_hits_top5": sum(1 for hit in row.get("top_hits", []) if hit.get("is_relevant")),
                }
            items.append(
                {
                    "query_id": query_id,
                    "doc_id": base.get("doc_id"),
                    "question": base.get("question"),
                    "gold_block_count": len(base.get("gold_block_ids") or []),
                    "case_type": self._case_type(by_strategy),
                    "strategies": strategies,
                }
            )
        return {"retriever": retriever, "queries": items}

    @lru_cache(maxsize=64)
    def bad_cases(self, retriever: str = "bm25") -> dict[str, Any]:
        queries = self.queries(retriever)["queries"]
        bad = [
            item
            for item in queries
            if item["case_type"] in {"hsc_missing", "baseline_beats_hsc", "strategy_disagreement"}
        ]
        bad.sort(
            key=lambda item: (
                0 if item["case_type"] == "hsc_missing" else 1,
                item["strategies"].get("hsc_rag", {}).get("recall@5", 0.0),
                item["question"] or "",
            )
        )
        return {"retriever": retriever, "queries": bad}

    @lru_cache(maxsize=256)
    def query_comparison(self, query_id: str, retriever: str = "bm25") -> dict[str, Any]:
        by_strategy = self.query_index(retriever).get(query_id)
        if not by_strategy:
            return {"query_id": query_id, "retriever": retriever, "found": False}

        base = self._base_query_record(by_strategy)
        strategies: dict[str, Any] = {}
        for strategy in STRATEGIES:
            row = by_strategy.get(strategy)
            if not row:
                continue
            top_hits = []
            for hit in row.get("top_hits", [])[:5]:
                top_hits.append(
                    {
                        "rank": hit.get("rank"),
                        "chunk_id": hit.get("chunk_id"),
                        "score": hit.get("score"),
                        "is_relevant": bool(hit.get("is_relevant")),
                        "covered_gold_block_ids": hit.get("covered_gold_block_ids", []),
                        "title_path": hit.get("title_path", []),
                        "source_blocks": hit.get("source_blocks", []),
                        "token_count": hit.get("token_count"),
                        "quality_flags": hit.get("quality_flags", []),
                        "preview": hit.get("preview"),
                    }
                )
            strategies[strategy] = {
                "first_relevant_rank": row.get("first_relevant_rank", 0),
                "reciprocal_rank": row.get("reciprocal_rank", 0.0),
                "recall_by_k": row.get("recall_by_k", {}),
                "hit_by_k": row.get("hit_by_k", {}),
                "full_recall_by_k": row.get("full_recall_by_k", {}),
                "covered_gold_blocks_by_k": row.get("covered_gold_blocks_by_k", {}),
                "missing_gold_blocks_at_max_k": row.get("missing_gold_blocks_at_max_k", []),
                "ndcg@5": row.get("ndcg@5", 0.0),
                "top_hits": top_hits,
            }

        return {
            "found": True,
            "retriever": retriever,
            "query_id": query_id,
            "doc_id": base.get("doc_id"),
            "question": base.get("question"),
            "gold_block_ids": base.get("gold_block_ids", []),
            "case_type": self._case_type(by_strategy),
            "strategies": strategies,
        }

    def clear_cache(self) -> None:
        self.overview.cache_clear()
        self.metrics.cache_clear()
        self.query_index.cache_clear()
        self.queries.cache_clear()
        self.bad_cases.cache_clear()
        self.query_comparison.cache_clear()

    def _base_query_record(self, by_strategy: dict[str, dict[str, Any]]) -> dict[str, Any]:
        for strategy in STRATEGIES:
            if strategy in by_strategy:
                return by_strategy[strategy]
        return next(iter(by_strategy.values()))

    def _case_type(self, by_strategy: dict[str, dict[str, Any]]) -> str:
        hsc = by_strategy.get("hsc_rag")
        fixed = by_strategy.get("fixed")
        if not hsc:
            return "missing_hsc"
        hsc_missing = hsc.get("missing_gold_blocks_at_max_k") or []
        if hsc_missing:
            return "hsc_missing"
        if fixed and self._rank_value(fixed) < self._rank_value(hsc):
            return "baseline_beats_hsc"
        recall_values = {
            strategy: (row.get("recall_by_k") or {}).get("5", 0.0)
            for strategy, row in by_strategy.items()
        }
        if len(set(recall_values.values())) > 1:
            return "strategy_disagreement"
        return "hsc_ok"

    def _rank_value(self, row: dict[str, Any]) -> int:
        rank = row.get("first_relevant_rank") or 0
        return rank if rank > 0 else 10_000

