# -*- coding: utf-8 -*-
"""Run the LangChain-backed HSC-RAG agent from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.chunkers.common import load_governed_documents  # noqa: E402
from app.core.schemas import LangChainAgentRequest  # noqa: E402
from app.services.langchain_agent_service import run_langchain_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "qasper" / "train" / "governed_documents.jsonl"),
        help="GovernedDocument JSONL input.",
    )
    parser.add_argument("--limit-docs", type=int, default=1, help="Number of documents to pass to the agent.")
    parser.add_argument(
        "--instruction",
        default="Use LangChain tools to chunk the governed document for RAG.",
        help="Natural-language instruction sent to the agent.",
    )
    parser.add_argument("--strategy", default="hsc_rag", choices=["fixed", "recursive", "semantic", "hsc_rag"])
    parser.add_argument("--config-json", default="{}", help="Chunker config as JSON object.")
    parser.add_argument("--preferred-tool", default=None)
    parser.add_argument("--provider", default="mock", choices=["mock", "openai_compatible"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()
    config = _parse_config(args.config_json)
    docs = load_governed_documents(Path(args.input))
    if args.limit_docs > 0:
        docs = docs[: args.limit_docs]

    request = LangChainAgentRequest(
        instruction=args.instruction,
        documents=docs,
        strategy=args.strategy,
        config=config,
        preferred_tool=args.preferred_tool,
        llm_provider=args.provider,
        llm_model=args.model,
        llm_base_url=args.base_url,
        llm_api_key_env=args.api_key_env,
        llm_temperature=args.temperature,
        llm_timeout_seconds=args.timeout_seconds,
    )
    response = run_langchain_agent(request)
    payload = response.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _parse_config(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--config-json must decode to a JSON object.")
    return value


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
