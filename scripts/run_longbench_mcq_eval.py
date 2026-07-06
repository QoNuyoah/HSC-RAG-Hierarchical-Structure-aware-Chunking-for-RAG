# -*- coding: utf-8 -*-
"""Evaluate LongBench-v2 multiple-choice answering over retrieved chunks.

This is an advanced task-level evaluation supplement. It does not replace
evidence Recall/nDCG because LongBench-v2 has no official evidence spans.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from app.llm.providers import LlmMessage, build_json_provider  # noqa: E402
from run_retrieval_eval import build_retriever, read_jsonl  # noqa: E402


WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "according",
    "through",
    "which",
    "what",
    "when",
    "where",
    "how",
    "why",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "use",
    "uses",
    "using",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governed-documents", required=True, help="LongBench governed_documents.jsonl.")
    parser.add_argument("--chunk-dir", required=True, help="Directory containing chunks_{strategy}.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output directory for MCQ predictions and summary.")
    parser.add_argument("--strategies", default="fixed,hsc_rag")
    parser.add_argument("--retrievers", default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--dense-encoder", default="tfidf_svd", choices=["tfidf_svd", "sentence_transformer", "auto"])
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dense-svd-dim", type=int, default=128)
    parser.add_argument("--hybrid-alpha", type=float, default=0.55)
    parser.add_argument(
        "--tokenizer-profile",
        default="mixed",
        choices=["mixed", "cjk_bigram", "cjk_2_4gram", "jieba"],
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--judge-provider",
        default="lexical",
        choices=["lexical", "mock", "openai_compatible"],
        help="lexical is deterministic offline. mock uses the LLM provider fallback contract.",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--llm-max-output-tokens", type=int, default=300)
    parser.add_argument("--llm-disable-response-format", action="store_true")
    parser.add_argument("--llm-fail-on-provider-error", action="store_true")
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_longbench_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for doc in read_jsonl(path):
        for query in doc.get("queries") or []:
            metadata = query.get("metadata") or {}
            choices = metadata.get("choices") or {}
            answer_label = normalize_answer_label(metadata.get("answer_label") or query.get("answer"))
            if not choices or answer_label not in {"A", "B", "C", "D"}:
                continue
            queries.append(
                {
                    "query_id": query.get("query_id"),
                    "doc_id": query.get("doc_id"),
                    "question": query.get("question") or "",
                    "answer_label": answer_label,
                    "choices": {label: str(choices.get(label) or "") for label in ["A", "B", "C", "D"]},
                    "metadata": metadata,
                }
            )
    return queries


def normalize_answer_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else ""


def content_terms(text: str) -> set[str]:
    return {term.lower() for term in WORD_RE.findall(text or "") if term.lower() not in STOPWORDS}


def lexical_judge(
    *,
    question: str,
    choices: dict[str, str],
    retrieved_text: str,
) -> dict[str, Any]:
    context_lower = retrieved_text.lower()
    question_terms = content_terms(question)
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for label, choice in choices.items():
        choice_terms = content_terms(choice)
        matched_choice = sorted(choice_terms & content_terms(retrieved_text))
        matched_question = sorted(question_terms & content_terms(retrieved_text))
        exact = bool(choice and choice.lower() in context_lower)
        choice_overlap = len(matched_choice) / max(1, len(choice_terms))
        question_overlap = len(matched_question) / max(1, len(question_terms))
        score = round((2.0 if exact else 0.0) + choice_overlap + 0.15 * question_overlap, 6)
        scored.append(
            (
                score,
                label,
                {
                    "label": label,
                    "score": score,
                    "exact_choice_text_found": exact,
                    "matched_choice_terms": matched_choice[:30],
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_label, _detail = scored[0]
    return {
        "answer": best_label,
        "confidence": min(1.0, round(best_score / 3.0, 4)),
        "rationale": "Deterministic lexical choice-overlap judge over retrieved chunks.",
        "choice_scores": [detail for _score, _label, detail in scored],
        "provider_execution": "lexical_offline_judge",
    }


def llm_judge(
    *,
    provider,
    question: str,
    choices: dict[str, str],
    retrieved_text: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    choice_lines = "\n".join(f"{label}. {choices.get(label, '')}" for label in ["A", "B", "C", "D"])
    context = retrieved_text[:12000]
    messages = [
        LlmMessage(
            role="system",
            content=(
                "You answer LongBench multiple-choice questions using only the provided retrieved context. "
                "Return JSON with answer as one of A, B, C, D, confidence from 0 to 1, and a short rationale."
            ),
        ),
        LlmMessage(
            role="user",
            content=(
                f"Question:\n{question}\n\nChoices:\n{choice_lines}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Return JSON only: {\"answer\":\"A|B|C|D\",\"confidence\":0.0,\"rationale\":\"...\"}"
            ),
        ),
    ]
    result = provider.complete_json(messages, fallback=fallback)
    answer = normalize_answer_label(result.get("answer"))
    if answer not in {"A", "B", "C", "D"}:
        result["answer"] = fallback["answer"]
        result["provider_execution"] = f"{result.get('provider_execution', 'provider')}_invalid_answer_fallback"
    else:
        result["answer"] = answer
    return result


def evaluate_strategy_retriever(
    *,
    strategy: str,
    retriever_name: str,
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    args: argparse.Namespace,
    provider,
    output_dir: Path,
) -> dict[str, Any]:
    retriever = build_retriever(
        retriever_name=retriever_name,
        chunks=chunks,
        include_metadata=args.include_metadata,
        dense_encoder=args.dense_encoder,
        dense_model=args.dense_model,
        dense_svd_dim=args.dense_svd_dim,
        hybrid_alpha=args.hybrid_alpha,
        local_files_only=not args.allow_model_download,
        tokenizer_profile=args.tokenizer_profile,
    )
    chunk_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}
    prediction_path = output_dir / f"longbench_mcq_predictions_{strategy}_{retriever_name}.jsonl"
    correct = 0
    evaluated = 0
    provider_counts: Counter[str] = Counter()

    with prediction_path.open("w", encoding="utf-8") as file:
        for query in queries:
            hits = retriever.search(
                query["question"],
                top_k=args.top_k,
                doc_id=None if args.global_search else query["doc_id"],
            )
            hit_chunks = [chunk_by_id.get(hit.chunk_id) for hit in hits]
            hit_chunks = [chunk for chunk in hit_chunks if chunk]
            retrieved_text = "\n\n".join(str(chunk.get("text") or "") for chunk in hit_chunks)
            fallback = lexical_judge(
                question=query["question"],
                choices=query["choices"],
                retrieved_text=retrieved_text,
            )
            if args.judge_provider == "lexical":
                judged = fallback
            else:
                judged = llm_judge(
                    provider=provider,
                    question=query["question"],
                    choices=query["choices"],
                    retrieved_text=retrieved_text,
                    fallback=fallback,
                )
            predicted = normalize_answer_label(judged.get("answer"))
            is_correct = predicted == query["answer_label"]
            correct += int(is_correct)
            evaluated += 1
            provider_counts[str(judged.get("provider_execution", args.judge_provider))] += 1
            record = {
                "query_id": query["query_id"],
                "doc_id": query["doc_id"],
                "strategy": strategy,
                "retriever": retriever_name,
                "top_k": args.top_k,
                "question": query["question"],
                "choices": query["choices"],
                "gold_answer": query["answer_label"],
                "predicted_answer": predicted,
                "correct": is_correct,
                "judge": judged,
                "hits": [hit.to_dict() for hit in hits],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct / evaluated if evaluated else 0.0
    return {
        "strategy": strategy,
        "retriever": retriever_name,
        "chunks": len(chunks),
        "queries_evaluated": evaluated,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "prediction_path": display_path(prediction_path),
        "retriever_config": retriever.config(),
        "provider_execution_counts": dict(sorted(provider_counts.items())),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# LongBench-v2 MCQ Evaluation",
        "",
        f"- governed_documents: `{summary['governed_documents']}`",
        f"- chunk_dir: `{summary['chunk_dir']}`",
        f"- judge_provider: `{summary['judge_provider']}`",
        f"- top_k: `{summary['top_k']}`",
        f"- global_search: `{summary['global_search']}`",
        f"- queries_evaluated: `{summary['queries_evaluated']}`",
        "",
        "| strategy | retriever | chunks | accuracy | correct / total |",
        "|---|---:|---:|---:|---:|",
    ]
    for report in summary["reports"]:
        lines.append(
            f"| {report['strategy']} | {report['retriever']} | {report['chunks']} | "
            f"{report['accuracy']:.4f} | {report['correct']} / {report['queries_evaluated']} |"
        )
    lines.extend(
        [
            "",
            "Note: this is a task-level multiple-choice evaluation over retrieved chunks. "
            "LongBench-v2 does not provide official evidence spans, so this supplements but "
            "does not replace evidence Recall/nDCG.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = load_longbench_queries(Path(args.governed_documents))

    provider = None
    if args.judge_provider in {"mock", "openai_compatible"}:
        provider = build_json_provider(
            provider=args.judge_provider,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_output_tokens,
            use_response_format=not args.llm_disable_response_format,
            timeout_seconds=args.llm_timeout_seconds,
            fallback_on_error=not args.llm_fail_on_provider_error,
        )

    reports: list[dict[str, Any]] = []
    chunk_dir = Path(args.chunk_dir)
    for strategy in parse_csv(args.strategies):
        chunks_path = chunk_dir / f"chunks_{strategy}.jsonl"
        if not chunks_path.exists():
            continue
        chunks = read_jsonl(chunks_path)
        for retriever_name in parse_csv(args.retrievers):
            reports.append(
                evaluate_strategy_retriever(
                    strategy=strategy,
                    retriever_name=retriever_name,
                    chunks=chunks,
                    queries=queries,
                    args=args,
                    provider=provider,
                    output_dir=output_dir,
                )
            )

    summary = {
        "task": "longbench_v2_multiple_choice_over_retrieved_chunks",
        "governed_documents": display_path(Path(args.governed_documents)),
        "chunk_dir": display_path(chunk_dir),
        "output_dir": display_path(output_dir),
        "strategies": parse_csv(args.strategies),
        "retrievers": parse_csv(args.retrievers),
        "top_k": args.top_k,
        "global_search": args.global_search,
        "judge_provider": args.judge_provider,
        "queries_evaluated": len(queries),
        "reports": reports,
        "note": (
            "LongBench-v2 has answer labels but no official evidence spans. "
            "MCQ accuracy is an advanced task-level supplement over retrieved chunks."
        ),
    }
    (output_dir / "longbench_mcq_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "longbench_mcq_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
