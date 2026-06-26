# -*- coding: utf-8 -*-
"""Run LLM-assisted semantic enrichment over HSC-RAG chunks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.schemas import RagChunk  # noqa: E402
from app.llm.chunk_enricher import ChunkEnrichmentConfig, ChunkSemanticEnricher  # noqa: E402
from app.llm.providers import build_json_provider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "qasper" / "train" / "chunks_hsc_rag.jsonl"),
        help="Input chunks JSONL. Defaults to QASPER HSC-RAG chunks.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "qasper" / "train" / "chunks_hsc_rag_llm_enriched.jsonl"),
        help="Output enriched chunks JSONL.",
    )
    parser.add_argument(
        "--report-md",
        default=str(PROJECT_ROOT / "reports" / "llm_enrichment_summary.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--report-json",
        default=str(PROJECT_ROOT / "reports" / "llm_enrichment_summary.json"),
        help="Machine-readable report path.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of chunks to enrich. Use 0 for all chunks.")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai_compatible"])
    parser.add_argument("--model", default=None, help="Model name for the selected provider.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable that stores the API key.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=700, help="Max tokens requested from the LLM provider.")
    parser.add_argument("--max-input-chars", type=int, default=2200, help="Max chunk characters sent to the LLM provider.")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send OpenAI JSON response_format. Useful for providers/models that do not support it.",
    )
    parser.add_argument(
        "--fail-on-provider-error",
        action="store_true",
        help="Raise provider errors instead of writing deterministic fallback results.",
    )
    parser.add_argument(
        "--include-qa",
        action="store_true",
        help="Also ask for one QA/instruction sample per enriched chunk.",
    )
    parser.add_argument(
        "--qa-output",
        default=None,
        help="Optional JSONL path for extracted QA/instruction samples.",
    )
    return parser.parse_args()


def load_chunks(path: Path, limit: int) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(RagChunk.model_validate(json.loads(line)))
            if limit > 0 and len(chunks) >= limit:
                break
    return chunks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_qa_pairs(chunks: list[RagChunk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        enrichment = (chunk.metadata or {}).get("llm_enrichment", {})
        for index, pair in enumerate(enrichment.get("qa_pairs", []), start=1):
            rows.append(
                {
                    "qa_id": f"{chunk.chunk_id}_qa_{index:02d}",
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "dataset": chunk.dataset,
                    "split": chunk.split,
                    "instruction": pair.get("instruction", ""),
                    "question": pair.get("question", ""),
                    "answer": pair.get("answer", ""),
                    "answerability": pair.get("answerability", "unknown"),
                    "faithfulness_score": pair.get("faithfulness_score"),
                    "evidence_source_blocks": pair.get("evidence_source_blocks", chunk.source_blocks[:5]),
                    "prompt_version": enrichment.get("prompt_version"),
                    "provider": enrichment.get("provider"),
                    "model": enrichment.get("model"),
                }
            )
    return rows


def build_report(
    *,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    report_json_path: Path,
    qa_output_path: Path | None,
    chunks: list[RagChunk],
    qa_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    enrichments = [(chunk.metadata or {}).get("llm_enrichment", {}) for chunk in chunks]
    risk_counts = Counter(item.get("faithfulness_risk", "unknown") for item in enrichments)
    provider_execution_counts = Counter(item.get("provider_execution", "unknown") for item in enrichments)
    avg = _score_average(enrichments)
    report = {
        "task": "HSC-RAG LLM semantic organization",
        "input": display_path(input_path),
        "output": display_path(output_path),
        "qa_output": display_path(qa_output_path) if qa_output_path else None,
        "report_json": display_path(report_json_path),
        "provider": args.provider,
        "model": args.model or ("mock-semantic-organizer-v1" if args.provider == "mock" else None),
        "limit": args.limit,
        "chunks": len(chunks),
        "qa_pairs": len(qa_rows),
        "prompt_version": "hsc-rag-enrich-v1",
        "score_average": avg,
        "faithfulness_risk_counts": dict(sorted(risk_counts.items())),
        "provider_execution_counts": dict(sorted(provider_execution_counts.items())),
        "sample_chunk_ids": [chunk.chunk_id for chunk in chunks[:5]],
    }
    return report


def write_markdown_report(path: Path, report: dict[str, Any], chunks: list[RagChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for chunk in chunks[:5]:
        enrichment = (chunk.metadata or {}).get("llm_enrichment", {})
        rows.append(
            "| {chunk_id} | {risk} | {integrity} | {faithfulness} | {tags} | {summary} |".format(
                chunk_id=_md(chunk.chunk_id),
                risk=_md(str(enrichment.get("faithfulness_risk", ""))),
                integrity=enrichment.get("semantic_integrity_score", ""),
                faithfulness=enrichment.get("summary_faithfulness_score", ""),
                tags=_md(", ".join(enrichment.get("topic_tags", [])[:4])),
                summary=_md(_truncate(str(enrichment.get("summary", "")), 110)),
            )
        )
    content = f"""# HSC-RAG 大模型语义组织增强报告

## 目的

本报告用于补充课题 11“面向 RAG 的智能分段与内容组织智能体”中的大模型相关能力。HSC-RAG 的分段边界仍由确定性结构感知算法完成，原因是边界控制、长度约束、原文回链和表格/公式保护更需要稳定、可追溯、可回放；大模型用于分段后的语义组织 Skill，包括 chunk 摘要、主题标签、实体标签、语义完整性评价、摘要忠实度评价和可选 QA/指令数据合成。

## 运行配置

| 项目 | 值 |
|---|---|
| 输入文件 | `{report["input"]}` |
| 输出文件 | `{report["output"]}` |
| QA 输出 | `{report.get("qa_output") or "未启用"}` |
| Provider | `{report["provider"]}` |
| Model | `{report.get("model") or "未设置"}` |
| Prompt Version | `{report["prompt_version"]}` |
| 处理 chunk 数 | {report["chunks"]} |
| QA/指令样例数 | {report["qa_pairs"]} |

## 指标摘要

| 指标 | 均值 |
|---|---:|
| 语义完整性评分 /5 | {report["score_average"].get("semantic_integrity_score", 0):.2f} |
| 摘要忠实度评分 /5 | {report["score_average"].get("summary_faithfulness_score", 0):.2f} |
| 标签准确性评分 /5 | {report["score_average"].get("tag_accuracy_score", 0):.2f} |

faithfulness risk 分布：

```json
{json.dumps(report["faithfulness_risk_counts"], ensure_ascii=False, indent=2)}
```

Provider execution 分布：

```json
{json.dumps(report["provider_execution_counts"], ensure_ascii=False, indent=2)}
```

## 样例

| chunk_id | 风险 | 语义完整 | 摘要忠实 | 标签 | 摘要 |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

## 与任务书要求的对应关系

- 内容打标：输出 `metadata.llm_enrichment.topic_tags`，作为大模型辅助主题/关键词标签。
- 摘要生成：输出 `metadata.llm_enrichment.summary`，并给出 `summary_faithfulness_score`。
- 实体标签：输出 `metadata.llm_enrichment.entity_tags`，用于补充规则实体标签。
- 语义完整：输出 `semantic_integrity_score` 和 `quality_reason`，用于人工抽样评价和 bad case 分析。
- 可私有化部署：`mock` 模式可离线复现；`openai_compatible` 模式后续可接入外部或本地兼容大模型服务。
- 不替代核心分段：大模型只负责语义组织，不负责不可控地改写分段边界，保证 chunk 的来源锚点和治理链路可追溯。
"""
    path.write_text(content, encoding="utf-8")


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _score_average(enrichments: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["semantic_integrity_score", "summary_faithfulness_score", "tag_accuracy_score"]
    result: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in enrichments if isinstance(item.get(key), (int, float))]
        result[key] = round(mean(values), 4) if values else 0.0
    return result


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_md_path = Path(args.report_md)
    report_json_path = Path(args.report_json)
    qa_output_path = Path(args.qa_output) if args.qa_output else None
    if args.include_qa and qa_output_path is None:
        qa_output_path = output_path.with_name("hsc_rag_synthetic_qa.jsonl")

    provider = build_json_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        use_response_format=not args.disable_response_format,
        timeout_seconds=args.timeout_seconds,
        fallback_on_error=not args.fail_on_provider_error,
    )
    enricher = ChunkSemanticEnricher(
        provider=provider,
        config=ChunkEnrichmentConfig(
            max_input_chars=args.max_input_chars,
            include_qa=args.include_qa,
        ),
    )

    chunks = load_chunks(input_path, limit=args.limit)
    enriched_chunks = [enricher.enrich_chunk(chunk) for chunk in chunks]
    write_jsonl(output_path, [chunk.model_dump(mode="json") for chunk in enriched_chunks])

    qa_rows = extract_qa_pairs(enriched_chunks)
    if qa_output_path is not None:
        write_jsonl(qa_output_path, qa_rows)

    report = build_report(
        args=args,
        input_path=input_path,
        output_path=output_path,
        report_json_path=report_json_path,
        qa_output_path=qa_output_path,
        chunks=enriched_chunks,
        qa_rows=qa_rows,
    )
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report_md_path, report, enriched_chunks)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
