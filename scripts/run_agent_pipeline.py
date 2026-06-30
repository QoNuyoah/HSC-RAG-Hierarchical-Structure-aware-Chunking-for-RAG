# -*- coding: utf-8 -*-
"""Unified runnable HSC-RAG agent pipeline.

This script is the product-style entry point:
input document(s) -> chunking -> optional retrieval evaluation ->
optional LLM semantic organization -> pipeline report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.chunkers.common import estimate_tokens, load_governed_documents, normalize_text, write_jsonl  # noqa: E402
from app.core.schemas import GovernedBlock, GovernedDocument, RagChunk, SourceAnchor  # noqa: E402
from app.llm.chunk_enricher import ChunkEnrichmentConfig, ChunkSemanticEnricher  # noqa: E402
from app.llm.providers import build_json_provider  # noqa: E402
from run_chunking import report_for, run_strategy  # noqa: E402
from run_retrieval_eval import (  # noqa: E402
    best_by_metric,
    build_retriever,
    compare_against_fixed,
    evaluate_pair,
    gold_records,
    parse_top_k,
    read_jsonl,
)
from run_llm_enrichment import build_report as build_llm_report  # noqa: E402
from run_llm_enrichment import extract_qa_pairs, write_markdown_report as write_llm_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", required=True, help="Input file: governed JSONL/JSON or Markdown.")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "governed_jsonl", "governed_json", "markdown"],
        help="Input format. auto infers from file suffix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline output directory. Defaults to runs/hsc_rag_agent_<timestamp>.",
    )
    parser.add_argument("--dataset", default="user_supplied", help="Dataset name for Markdown/JSON normalization.")
    parser.add_argument("--split", default="demo", help="Split name for Markdown/JSON normalization.")
    parser.add_argument("--doc-id", default=None, help="Document id for Markdown input.")

    parser.add_argument("--strategies", default="fixed,hsc_rag", help="Comma-separated chunk strategies.")
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
    parser.add_argument("--hsc-boundary-threshold", type=float, default=0.62)
    parser.add_argument("--hsc-soft-boundary-threshold", type=float, default=0.52)
    parser.add_argument("--hsc-semantic-distance-threshold", type=float, default=0.72)
    parser.add_argument("--hsc-semantic-window-blocks", type=int, default=3)

    parser.add_argument("--run-eval", action="store_true", help="Run retrieval evaluation after chunking.")
    parser.add_argument("--gold-evidence", default=None, help="gold_evidence.jsonl for retrieval evaluation.")
    parser.add_argument("--retrievers", default="bm25,dense,hybrid")
    parser.add_argument("--top-k", default="1,3,5")
    parser.add_argument("--ndcg-k", type=int, default=5)
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--dense-encoder", default="tfidf_svd", choices=["tfidf_svd", "sentence_transformer", "auto"])
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-svd-dim", type=int, default=128)
    parser.add_argument("--hybrid-alpha", type=float, default=0.55)
    parser.add_argument("--allow-model-download", action="store_true")

    parser.add_argument("--run-llm-enrichment", action="store_true", help="Run LLM semantic organization.")
    parser.add_argument("--llm-strategy", default="hsc_rag", help="Chunk strategy to enrich.")
    parser.add_argument("--llm-limit", type=int, default=20)
    parser.add_argument("--llm-provider", default="mock", choices=["mock", "openai_compatible"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-max-input-chars", type=int, default=2200)
    parser.add_argument("--llm-max-output-tokens", type=int, default=700)
    parser.add_argument("--llm-disable-response-format", action="store_true")
    parser.add_argument("--llm-fail-on-provider-error", action="store_true")
    parser.add_argument("--llm-include-qa", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    docs = load_input_documents(
        input_path=input_path,
        input_format=args.input_format,
        dataset=args.dataset,
        split=args.split,
        doc_id=args.doc_id,
    )
    governed_path = output_dir / "governed_documents.jsonl"
    write_jsonl(governed_path, [doc.model_dump(mode="json") for doc in docs])
    steps.append(
        {
            "step": "load_input",
            "status": "completed",
            "input": display_path(input_path),
            "input_format": infer_format(input_path, args.input_format),
            "documents": len(docs),
            "governed_documents": display_path(governed_path),
        }
    )

    chunk_reports = run_chunk_stage(
        docs=docs,
        input_path=governed_path,
        output_dir=output_dir,
        args=args,
    )
    steps.append(
        {
            "step": "chunking",
            "status": "completed",
            "strategies": parse_csv(args.strategies),
            "summary": display_path(output_dir / "chunking_summary.json"),
            "reports": chunk_reports,
        }
    )

    eval_summary: dict[str, Any] | None = None
    if args.run_eval:
        if not args.gold_evidence:
            steps.append(
                {
                    "step": "retrieval_evaluation",
                    "status": "skipped",
                    "reason": "--gold-evidence is required for Recall/MRR/nDCG evaluation.",
                }
            )
        else:
            eval_summary = run_eval_stage(output_dir=output_dir, args=args)
            steps.append(
                {
                    "step": "retrieval_evaluation",
                    "status": "completed",
                    "gold_evidence": display_path(Path(args.gold_evidence)),
                    "summary": display_path(output_dir / "retrieval_eval_multi_summary.json"),
                    "retrievers": parse_csv(args.retrievers),
                }
            )

    llm_summary: dict[str, Any] | None = None
    if args.run_llm_enrichment:
        llm_summary = run_llm_stage(output_dir=output_dir, args=args)
        steps.append(
            {
                "step": "llm_semantic_organization",
                "status": "completed",
                "summary": display_path(output_dir / "llm_enrichment_summary.json"),
                "markdown": display_path(output_dir / "llm_enrichment_summary.md"),
            }
        )

    pipeline_summary = {
        "agent": "hsc-rag-agent",
        "schema_version": "hsc-rag-agent-pipeline-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": display_path(input_path),
        "output_dir": display_path(output_dir),
        "documents": len(docs),
        "steps": steps,
        "chunking": {"reports": chunk_reports},
        "retrieval_evaluation": eval_summary,
        "llm_enrichment": llm_summary,
        "notes": [
            "Retrieval metrics require gold_evidence.jsonl with query/evidence labels.",
            "Markdown input is normalized into GovernedDocument before chunking.",
            "HSC-RAG does post-normalization packaging; upstream parsing/cleaning is outside this agent boundary.",
        ],
    }
    summary_path = output_dir / "agent_pipeline_summary.json"
    summary_path.write_text(json.dumps(pipeline_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_pipeline_markdown(output_dir / "agent_pipeline_summary.md", pipeline_summary)
    print(json.dumps(pipeline_summary, ensure_ascii=False, indent=2))


def run_chunk_stage(
    *,
    docs: list[GovernedDocument],
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    chunk_args = SimpleNamespace(
        fixed_target=args.fixed_target,
        fixed_overlap=args.fixed_overlap,
        recursive_target=args.recursive_target,
        recursive_overlap=args.recursive_overlap,
        semantic_min=args.semantic_min,
        semantic_target=args.semantic_target,
        semantic_max=args.semantic_max,
        semantic_breakpoint_percentile=args.semantic_breakpoint_percentile,
        hsc_min=args.hsc_min,
        hsc_target=args.hsc_target,
        hsc_max=args.hsc_max,
        hsc_boundary_threshold=args.hsc_boundary_threshold,
        hsc_soft_boundary_threshold=args.hsc_soft_boundary_threshold,
        hsc_semantic_distance_threshold=args.hsc_semantic_distance_threshold,
        hsc_semantic_window_blocks=args.hsc_semantic_window_blocks,
    )
    reports: list[dict[str, Any]] = []
    for strategy in parse_csv(args.strategies):
        chunks, config = run_strategy(strategy, docs, chunk_args)
        chunk_path = output_dir / f"chunks_{strategy}.jsonl"
        report_path = output_dir / f"chunk_report_{strategy}.json"
        write_jsonl(chunk_path, [chunk.model_dump(mode="json") for chunk in chunks])
        report = report_for(
            strategy=strategy,
            input_path=input_path,
            output_path=chunk_path,
            documents=len(docs),
            chunks=chunks,
            config=config,
        )
        report_json = report.model_dump(mode="json")
        report_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
        reports.append(report_json)

    (output_dir / "chunking_summary.json").write_text(
        json.dumps({"reports": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return reports


def run_eval_stage(*, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    top_ks = parse_top_k(args.top_k)
    records, skipped = gold_records(Path(args.gold_evidence))
    strategies = parse_csv(args.strategies)
    retriever_names = parse_csv(args.retrievers)
    doc_filter = not args.global_search
    local_files_only = not args.allow_model_download

    reports: list[dict[str, Any]] = []
    for strategy in strategies:
        chunks_path = output_dir / f"chunks_{strategy}.jsonl"
        if not chunks_path.exists():
            continue
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
        "gold_evidence_path": display_path(Path(args.gold_evidence)),
        "output_dir": display_path(output_dir),
        "strategies": strategies,
        "retrievers": retriever_names,
        "reports": reports,
        "comparisons_against_fixed": compare_against_fixed(reports),
        "best_by_metric": best_by_metric(reports),
    }
    (output_dir / "retrieval_eval_multi_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_llm_stage(*, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    input_path = output_dir / f"chunks_{args.llm_strategy}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing chunks for LLM enrichment: {input_path}")
    output_path = output_dir / f"chunks_{args.llm_strategy}_llm_enriched.jsonl"
    report_json_path = output_dir / "llm_enrichment_summary.json"
    report_md_path = output_dir / "llm_enrichment_summary.md"
    qa_output_path = output_dir / "hsc_rag_synthetic_qa.jsonl" if args.llm_include_qa else None

    provider = build_json_provider(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key_env=args.llm_api_key_env,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_output_tokens,
        use_response_format=not args.llm_disable_response_format,
        timeout_seconds=args.llm_timeout_seconds,
        fallback_on_error=not args.llm_fail_on_provider_error,
    )
    enricher = ChunkSemanticEnricher(
        provider=provider,
        config=ChunkEnrichmentConfig(
            max_input_chars=args.llm_max_input_chars,
            include_qa=args.llm_include_qa,
        ),
    )
    chunks = load_chunks(input_path, limit=args.llm_limit)
    enriched_chunks = [enricher.enrich_chunk(chunk) for chunk in chunks]
    write_jsonl(output_path, [chunk.model_dump(mode="json") for chunk in enriched_chunks])

    qa_rows = extract_qa_pairs(enriched_chunks)
    if qa_output_path is not None:
        write_jsonl(qa_output_path, qa_rows)

    llm_args = SimpleNamespace(
        provider=args.llm_provider,
        model=args.llm_model,
        limit=args.llm_limit,
    )
    report = build_llm_report(
        args=llm_args,
        input_path=input_path,
        output_path=output_path,
        report_json_path=report_json_path,
        qa_output_path=qa_output_path,
        chunks=enriched_chunks,
        qa_rows=qa_rows,
    )
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_llm_markdown(report_md_path, report, enriched_chunks)
    return report


def load_chunks(path: Path, limit: int) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for record in read_jsonl(path):
        chunks.append(RagChunk.model_validate(record))
        if limit > 0 and len(chunks) >= limit:
            break
    return chunks


def load_input_documents(
    *,
    input_path: Path,
    input_format: str,
    dataset: str,
    split: str,
    doc_id: str | None,
) -> list[GovernedDocument]:
    actual_format = infer_format(input_path, input_format)
    if actual_format == "governed_jsonl":
        return load_governed_documents(input_path)
    if actual_format == "governed_json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "documents" in data:
            return [GovernedDocument.model_validate(item) for item in data["documents"]]
        if isinstance(data, list):
            return [GovernedDocument.model_validate(item) for item in data]
        return [GovernedDocument.model_validate(data)]
    if actual_format == "markdown":
        return [markdown_to_governed_document(input_path, dataset=dataset, split=split, doc_id=doc_id)]
    raise ValueError(f"Unsupported input format: {actual_format}")


def markdown_to_governed_document(
    path: Path,
    *,
    dataset: str,
    split: str,
    doc_id: str | None,
) -> GovernedDocument:
    raw = path.read_text(encoding="utf-8")
    doc_id = doc_id or safe_id(path.stem)
    title = path.stem
    blocks: list[GovernedBlock] = []
    title_stack: list[str] = []
    order = 1

    def add_block(block_type: str, text: str, level: int = 0) -> None:
        nonlocal order
        text = normalize_text(text)
        if not text:
            return
        block_id = f"{doc_id}_b{order:05d}"
        section = title_stack[-1] if title_stack else title
        blocks.append(
            GovernedBlock(
                block_id=block_id,
                doc_id=doc_id,
                type=block_type,  # type: ignore[arg-type]
                text=text,
                order=order,
                level=level,
                title_path=list(title_stack),
                source_anchor=SourceAnchor(
                    dataset=dataset,
                    split=split,
                    source_doc_id=str(path),
                    section_name=section,
                    paragraph_index=order,
                    extra={"input_format": "markdown", "source_file": str(path)},
                ),
                entity_tags=derive_markdown_entities(text),
                metadata={"source_line_group": order},
            )
        )
        order += 1

    paragraph: list[str] = []
    table: list[str] = []
    list_items: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            add_block("paragraph", " ".join(paragraph), level=len(title_stack))
            paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            add_block("table", "\n".join(table), level=len(title_stack))
            table = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            add_block("list", "\n".join(list_items), level=len(title_stack))
            list_items = []

    for raw_line in raw.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            flush_table()
            flush_list()
            if in_code:
                add_block("code", "\n".join(code_lines), level=len(title_stack))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            flush_table()
            flush_list()
            level = len(heading.group(1))
            heading_text = normalize_text(heading.group(2))
            if level == 1 and not blocks:
                title = heading_text
            title_stack = title_stack[: level - 1] + [heading_text]
            add_block("heading", heading_text, level=level)
            continue
        if not stripped:
            flush_paragraph()
            flush_table()
            flush_list()
            continue
        if "|" in stripped and stripped.count("|") >= 2:
            flush_paragraph()
            flush_list()
            table.append(stripped)
            continue
        if re.match(r"^([-*+]|\d+[.)])\s+", stripped):
            flush_paragraph()
            flush_table()
            list_items.append(stripped)
            continue
        flush_table()
        flush_list()
        paragraph.append(stripped)

    flush_paragraph()
    flush_table()
    flush_list()
    if code_lines:
        add_block("code", "\n".join(code_lines), level=len(title_stack))

    return GovernedDocument(
        doc_id=doc_id,
        dataset=dataset,
        split=split,
        source_doc_id=str(path),
        title=title,
        normalization_status="simulated_governed",
        term_policy="markdown_input_assumed_governed",
        blocks=blocks,
        source_ref={"path": str(path), "input_format": "markdown"},
        metadata={
            "adapter": "markdown_to_governed_document",
            "warning": "Markdown is treated as already governed text; no OCR, cleaning, or terminology normalization is performed.",
        },
    )


def derive_markdown_entities(text: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_+\-]{2,}\b|[\u4e00-\u9fff]{2,8}", text):
        value = normalize_text(match.group(0))
        if value and value not in entities:
            entities.append(value)
        if len(entities) >= 12:
            break
    return entities


def infer_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".jsonl":
        return "governed_jsonl"
    if suffix == ".json":
        return "governed_json"
    raise ValueError(f"Cannot infer input format from suffix: {path}")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / f"hsc_rag_agent_{stamp}"


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_id(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")
    return safe or "document"


def write_pipeline_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# HSC-RAG Agent Pipeline Report",
        "",
        f"- Input: `{summary['input']}`",
        f"- Output directory: `{summary['output_dir']}`",
        f"- Documents: {summary['documents']}",
        "",
        "## Steps",
        "",
        "| Step | Status | Output |",
        "|---|---|---|",
    ]
    for step in summary["steps"]:
        output = step.get("summary") or step.get("governed_documents") or step.get("reason") or ""
        lines.append(f"| {step['step']} | {step['status']} | `{output}` |")

    lines.extend(["", "## Chunking"])
    for report in summary["chunking"]["reports"]:
        lines.append(
            "- `{strategy}`: chunks={chunks}, avg_tokens={avg_tokens}, max_tokens={max_tokens}, output=`{output_path}`".format(
                **report
            )
        )

    eval_summary = summary.get("retrieval_evaluation")
    if eval_summary:
        lines.extend(["", "## Retrieval Evaluation", "", "| Strategy | Retriever | R@1 | R@3 | R@5 | MRR | nDCG@5 |", "|---|---|---:|---:|---:|---:|---:|"])
        for report in eval_summary.get("reports", []):
            metrics = report["metrics"]
            lines.append(
                "| {strategy} | {retriever} | {r1:.6f} | {r3:.6f} | {r5:.6f} | {mrr:.6f} | {ndcg:.6f} |".format(
                    strategy=report["strategy"],
                    retriever=report["retriever"],
                    r1=metrics.get("recall@1", 0.0),
                    r3=metrics.get("recall@3", 0.0),
                    r5=metrics.get("recall@5", 0.0),
                    mrr=metrics.get("mrr", 0.0),
                    ndcg=metrics.get("ndcg@5", 0.0),
                )
            )

    llm_summary = summary.get("llm_enrichment")
    if llm_summary:
        lines.extend(
            [
                "",
                "## LLM Semantic Organization",
                "",
                f"- Provider: `{llm_summary.get('provider')}`",
                f"- Model: `{llm_summary.get('model')}`",
                f"- Chunks: {llm_summary.get('chunks')}",
                f"- Provider execution: `{json.dumps(llm_summary.get('provider_execution_counts', {}), ensure_ascii=False)}`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
