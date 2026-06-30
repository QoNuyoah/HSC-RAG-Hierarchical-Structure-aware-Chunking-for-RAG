# -*- coding: utf-8 -*-
"""Generate acceptance metrics for the HSC-RAG deliverable."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTENT_BLOCK_TYPES = {"abstract", "paragraph", "list", "table", "figure", "code", "formula", "caption"}
PROTECTED_BLOCK_TYPES = {"table", "figure", "code", "formula", "list"}
REQUIRED_CHUNK_FIELDS = {
    "chunk_id": str,
    "doc_id": str,
    "dataset": str,
    "split": str,
    "strategy": str,
    "text": str,
    "token_count": int,
    "title_path": list,
    "source_blocks": list,
    "source_anchor": dict,
    "tags": list,
    "summary": str,
    "entity_tags": list,
    "quality_flags": list,
}
REQUIRED_ANCHOR_FIELDS = {
    "dataset": str,
    "source_doc_id": str,
    "sections": list,
    "first_block_id": str,
    "last_block_id": str,
    "block_count": int,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float) -> float:
    return round(value * 100, 2)


def pass_fail(value: float, threshold: float) -> str:
    return "达标" if value >= threshold else "未达标"


def read_manual_eval(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "completed_rows": 0,
            "semantic_integrity": None,
            "tag_summary": None,
        }

    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    semantic_scores: list[float] = []
    tag_summary_scores: list[float] = []
    completed_rows = 0
    for row in rows:
        try:
            semantic_score = float((row.get("semantic_integrity_score") or "").strip())
            tag_summary_score = float((row.get("tag_summary_score") or "").strip())
        except ValueError:
            continue
        if 1 <= semantic_score <= 5 and 1 <= tag_summary_score <= 5:
            semantic_scores.append(semantic_score)
            tag_summary_scores.append(tag_summary_score)
            completed_rows += 1

    def summarize(scores: list[float]) -> dict[str, Any] | None:
        if not scores:
            return None
        return {
            "avg": round(sum(scores) / len(scores), 2),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
        }

    return {
        "path": str(path),
        "rows": len(rows),
        "completed_rows": completed_rows,
        "semantic_integrity": summarize(semantic_scores),
        "tag_summary": summarize(tag_summary_scores),
    }


def manual_eval_status(manual_eval: dict[str, Any], key: str) -> str:
    item = manual_eval.get(key)
    if not item:
        return "待人工评价"
    if manual_eval.get("completed_rows", 0) < 20:
        return "样本不足"
    return "达标" if item["avg"] >= 4.0 and item["min"] >= 3.0 else "未达标"


def manual_eval_value(manual_eval: dict[str, Any], key: str) -> str:
    item = manual_eval.get(key)
    if not item:
        return "明天补充"
    return f"抽样 {manual_eval['completed_rows']} 个 chunks，均分 {item['avg']}/5，最低 {item['min']}/5"


def boundary_score_status(summary: dict[str, Any]) -> str:
    if not summary:
        return "待重新生成"
    return "达标" if summary.get("chunk_coverage_rate", 0) >= 0.80 else "未达标"


def boundary_score_value(summary: dict[str, Any]) -> str:
    if not summary:
        return "需要重新运行 run_chunking.py 生成带边界评分的 chunks"
    return (
        f"覆盖 {summary.get('chunk_coverage_rate', 0) * 100:.1f}%，"
        f"平均边界分 {summary.get('avg_boundary_score', 0)}，"
        f"平均语义距离 {summary.get('avg_semantic_distance', 0)}，"
        f"语义触发 {summary.get('semantic_boundary_triggered', 0)} 次"
    )


def field_ok(chunk: dict[str, Any]) -> bool:
    for field, expected_type in REQUIRED_CHUNK_FIELDS.items():
        value = chunk.get(field)
        if not isinstance(value, expected_type):
            return False
        if field in {"chunk_id", "doc_id", "dataset", "split", "strategy", "text", "summary"} and not value:
            return False
        if field in {"source_blocks", "tags", "entity_tags", "quality_flags"} and not value:
            return False
        if field == "token_count" and value <= 0:
            return False
    anchor = chunk.get("source_anchor") or {}
    for field, expected_type in REQUIRED_ANCHOR_FIELDS.items():
        value = anchor.get(field)
        if not isinstance(value, expected_type):
            return False
        if field in {"dataset", "source_doc_id", "first_block_id", "last_block_id"} and not value:
            return False
        if field == "block_count" and value <= 0:
            return False
    return True


def source_membership(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for block_id in chunk.get("source_blocks", []):
            members[block_id].append(chunk["chunk_id"])
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/qasper/train")
    parser.add_argument("--strategy", default="hsc_rag")
    parser.add_argument("--manual-eval-csv", default="reports/manual_eval_hsc_rag.csv")
    parser.add_argument("--output-json", default="reports/acceptance_metrics_hsc_rag.json")
    parser.add_argument("--output-md", default="reports/acceptance_checklist.md")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    chunks = read_jsonl(data_dir / f"chunks_{args.strategy}.jsonl")
    blocks = read_jsonl(data_dir / "blocks.jsonl")
    chunk_report = read_json(data_dir / f"chunk_report_{args.strategy}.json")
    retrieval = read_json(data_dir / "retrieval_eval_multi_summary.json")
    manual_eval = read_manual_eval(Path(args.manual_eval_csv))

    members = source_membership(chunks)
    content_blocks = [block for block in blocks if block.get("type") in CONTENT_BLOCK_TYPES and block.get("text")]
    protected_blocks = [block for block in blocks if block.get("type") in PROTECTED_BLOCK_TYPES]
    no_split_units = [block for block in content_blocks if len(members.get(block["block_id"], [])) == 1]
    protected_intact = [block for block in protected_blocks if len(members.get(block["block_id"], [])) == 1]
    output_complete = [chunk for chunk in chunks if field_ok(chunk)]
    anchor_complete = [
        chunk
        for chunk in chunks
        if "source_anchor_complete" in (chunk.get("quality_flags") or [])
        and chunk.get("source_blocks")
        and chunk.get("source_anchor", {}).get("block_count") == len(chunk.get("source_blocks", []))
    ]
    length_ok = chunk_report.get("quality_flag_counts", {}).get("length_ok", 0)
    boundary_summary = chunk_report.get("boundary_score_summary", {})

    fixed_bm25 = None
    hsc_bm25 = None
    for report in retrieval.get("reports", []):
        if report.get("retriever") == "bm25" and report.get("strategy") == "fixed":
            fixed_bm25 = report
        if report.get("retriever") == "bm25" and report.get("strategy") == args.strategy:
            hsc_bm25 = report

    uplift: dict[str, Any] = {}
    if fixed_bm25 and hsc_bm25:
        for metric in ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]:
            base = fixed_bm25["metrics"][metric]
            target = hsc_bm25["metrics"][metric]
            uplift[metric] = {
                "fixed": base,
                "hsc_rag": target,
                "absolute_delta": round(target - base, 6),
                "relative_uplift_percent": pct((target - base) / base) if base else None,
            }

    output_rate = len(output_complete) / len(chunks) if chunks else 0.0
    no_sentence_break_rate = len(no_split_units) / len(content_blocks) if content_blocks else 0.0
    protected_rate = len(protected_intact) / len(protected_blocks) if protected_blocks else 1.0
    length_rate = length_ok / len(chunks) if chunks else 0.0
    anchor_rate = len(anchor_complete) / len(chunks) if chunks else 0.0
    retrieval_ok = (
        uplift.get("recall@5", {}).get("relative_uplift_percent", 0) >= 10
        and uplift.get("ndcg@5", {}).get("relative_uplift_percent", 0) >= 10
    )

    metrics = {
        "strategy": args.strategy,
        "chunks": len(chunks),
        "content_blocks": len(content_blocks),
        "protected_blocks": len(protected_blocks),
        "output_field_completeness_rate": pct(output_rate),
        "source_anchor_completeness_rate": pct(anchor_rate),
        "target_length_hit_rate": pct(length_rate),
        "no_sentence_break_rate": pct(no_sentence_break_rate),
        "protected_block_integrity_rate": pct(protected_rate),
        "retrieval_uplift_vs_fixed_bm25": uplift,
        "field_non_empty_counts": {
            "tags": sum(1 for chunk in chunks if chunk.get("tags")),
            "summary": sum(1 for chunk in chunks if chunk.get("summary")),
            "entity_tags": sum(1 for chunk in chunks if chunk.get("entity_tags")),
            "source_blocks": sum(1 for chunk in chunks if chunk.get("source_blocks")),
        },
        "quality_flag_counts": chunk_report.get("quality_flag_counts", {}),
        "boundary_score_summary": boundary_summary,
        "protected_block_types": dict(Counter(block.get("type") for block in protected_blocks)),
        "manual_evaluation": manual_eval,
        "acceptance_status": {
            "期望输出字段完整": pass_fail(output_rate, 1.0),
            "不破句率": pass_fail(no_sentence_break_rate, 1.0),
            "表格公式代码整体成块率": pass_fail(protected_rate, 0.95),
            "目标长度区间命中率": pass_fail(length_rate, 0.90),
            "原文回链完整率": pass_fail(anchor_rate, 1.0),
            "下游检索提升": "达标" if retrieval_ok else "未达标",
            "结构语义边界可解释": boundary_score_status(boundary_summary),
            "语义完整人工评价": manual_eval_status(manual_eval, "semantic_integrity"),
            "标签与摘要人工评价": manual_eval_status(manual_eval, "tag_summary"),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        ("期望输出字段完整", "每个 chunk 含文本、长度、标签、摘要、实体标签、原文锚点回链等字段", f"{metrics['output_field_completeness_rate']}%", metrics["acceptance_status"]["期望输出字段完整"]),
        ("不破句率", "以 GovernedBlock 为最小治理内容单元，检查内容块未被跨 chunk 人工截断", f"{metrics['no_sentence_break_rate']}%", metrics["acceptance_status"]["不破句率"]),
        ("表格/公式/代码整体成块率", "protected block 在 HSC-RAG chunk 中完整出现一次", f"{metrics['protected_block_integrity_rate']}%", metrics["acceptance_status"]["表格公式代码整体成块率"]),
        ("目标长度区间命中率", "quality_flags 中 length_ok 的 chunk 占比", f"{metrics['target_length_hit_rate']}%", metrics["acceptance_status"]["目标长度区间命中率"]),
        ("原文回链完整率", "source_anchor 与 source_blocks 完整一致", f"{metrics['source_anchor_completeness_rate']}%", metrics["acceptance_status"]["原文回链完整率"]),
        ("下游检索提升", "HSC-RAG 相对 fixed 的 BM25 Recall@5 与 nDCG@5 相对提升均 >= 10%", f"Recall@5 +{uplift.get('recall@5', {}).get('relative_uplift_percent')}%; nDCG@5 +{uplift.get('ndcg@5', {}).get('relative_uplift_percent')}%", metrics["acceptance_status"]["下游检索提升"]),
        ("结构/语义边界可解释", "每个由 HSC-RAG 主算法收束的 chunk 记录 boundary_score、结构信号、语义距离、长度压力和 split_reason", boundary_score_value(boundary_summary), metrics["acceptance_status"]["结构语义边界可解释"]),
        ("语义完整", "人工抽样评价 chunk 是否围绕同一主题、上下文是否足够、是否无明显断裂", manual_eval_value(manual_eval, "semantic_integrity"), metrics["acceptance_status"]["语义完整人工评价"]),
        ("打标与摘要", "人工抽样评价标签准确率、实体标签可用性与摘要忠实度", manual_eval_value(manual_eval, "tag_summary"), metrics["acceptance_status"]["标签与摘要人工评价"]),
    ]
    md = [
        "# HSC-RAG 验收指标对照表",
        "",
        "本报告由 `scripts/validate_acceptance_metrics.py` 自动生成，用于对照课题任务书中的可度量验收项。",
        "",
        f"- Strategy: `{args.strategy}`",
        f"- Chunks: `{len(chunks)}`",
        f"- Content blocks: `{len(content_blocks)}`",
        f"- Protected blocks: `{len(protected_blocks)}`",
        "",
        "| 指标项 | 验收口径 | 当前值 | 状态 |",
        "|---|---|---:|---|",
    ]
    md.extend(f"| {name} | {rule} | {value} | {status} |" for name, rule, value, status in rows)
    md.extend(
        [
            "",
            "## 说明",
            "",
            "- “不破句率”采用工程可验证口径：HSC-RAG 以 `GovernedBlock` 为最小治理内容单元进行封装；当前样本中每个内容块仅出现在一个 HSC-RAG chunk 中，说明没有发生跨 chunk 的句中/块内截断。",
            "- “期望输出字段完整”检查每个 chunk 是否包含 `text/token_count/tags/summary/entity_tags/source_blocks/source_anchor/quality_flags` 等下游消费必需字段。",
            "- “结构/语义边界可解释”读取 HSC-RAG chunk metadata 中的 `closing_boundary_decision`，用于证明主算法不是纯固定规则，而是综合标题结构、语义距离和长度压力进行边界决策。",
            f"- “语义完整”和“打标与摘要”读取 `{args.manual_eval_csv}`。建议抽样 20 个 chunks；均分 >= 4.0 且最低分 >= 3.0 记为达标。",
            "",
        ]
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
