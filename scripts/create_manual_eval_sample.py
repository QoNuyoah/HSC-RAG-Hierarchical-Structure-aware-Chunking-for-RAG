# -*- coding: utf-8 -*-
"""Create a reproducible manual-evaluation sample for HSC-RAG chunks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "sample_no",
    "chunk_id",
    "doc_id",
    "title_path",
    "token_count",
    "quality_flags",
    "tags",
    "entity_tags",
    "summary",
    "text_preview",
    "semantic_integrity_score",
    "semantic_integrity_comment",
    "tag_summary_score",
    "tag_summary_comment",
    "evaluator",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compact(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value or "")


def preview(text: str, limit: int = 900) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def even_sample(items: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size >= len(items):
        return items
    last = len(items) - 1
    indices = sorted({round(i * last / (sample_size - 1)) for i in range(sample_size)})
    return [items[index] for index in indices]


def write_csv(samples: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, chunk in enumerate(samples, start=1):
            writer.writerow(
                {
                    "sample_no": index,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title_path": compact(chunk.get("title_path")),
                    "token_count": chunk.get("token_count"),
                    "quality_flags": compact(chunk.get("quality_flags")),
                    "tags": compact(chunk.get("tags")),
                    "entity_tags": compact(chunk.get("entity_tags")),
                    "summary": chunk.get("summary", ""),
                    "text_preview": preview(chunk.get("text", "")),
                    "semantic_integrity_score": "",
                    "semantic_integrity_comment": "",
                    "tag_summary_score": "",
                    "tag_summary_comment": "",
                    "evaluator": "",
                }
            )


def write_markdown(samples: list[dict[str, Any]], output_path: Path, csv_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HSC-RAG 人工抽样评价表",
        "",
        f"- CSV 填分文件: `{csv_path}`",
        "- 抽样方式: 从 `chunks_hsc_rag.jsonl` 中按文档顺序均匀抽取 20 个 chunks。",
        "- 语义完整评分: 1-5 分，评价 chunk 是否围绕同一主题、上下文是否足够、是否没有明显断裂。",
        "- 打标与摘要评分: 1-5 分，评价 tags/entity_tags 是否有用、summary 是否忠实概括原文。",
        "- 达标口径: 20 个样本全部完成评分，均分 >= 4.0 且最低分 >= 3.0。",
        "",
    ]
    for index, chunk in enumerate(samples, start=1):
        lines.extend(
            [
                f"## 样本 {index}: {chunk['chunk_id']}",
                "",
                f"- Doc: `{chunk['doc_id']}`",
                f"- Title path: `{compact(chunk.get('title_path'))}`",
                f"- Token count: `{chunk.get('token_count')}`",
                f"- Quality flags: `{compact(chunk.get('quality_flags'))}`",
                f"- Tags: `{compact(chunk.get('tags'))}`",
                f"- Entity tags: `{compact(chunk.get('entity_tags'))}`",
                f"- Summary: {chunk.get('summary', '')}",
                "",
                "```text",
                chunk.get("text", ""),
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/processed/qasper/train/chunks_hsc_rag.jsonl")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--output-csv", default="reports/manual_eval_hsc_rag.csv")
    parser.add_argument("--output-md", default="reports/manual_eval_hsc_rag.md")
    args = parser.parse_args()

    chunks = read_jsonl(Path(args.chunks))
    samples = even_sample(chunks, args.sample_size)
    write_csv(samples, Path(args.output_csv))
    write_markdown(samples, Path(args.output_md), Path(args.output_csv))
    print(
        json.dumps(
            {
                "chunks": len(chunks),
                "sample_size": len(samples),
                "output_csv": args.output_csv,
                "output_md": args.output_md,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
