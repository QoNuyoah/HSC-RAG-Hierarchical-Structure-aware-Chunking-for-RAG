# -*- coding: utf-8 -*-
"""Validate the Topic 11 JSON handoff contract examples."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.schemas import ChunkAgentRequest, ChunkAgentResponse  # noqa: E402
from app.services.chunking_service import run_chunk_request  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        default=str(PROJECT_ROOT / "examples" / "topic11_request.json"),
        help="ChunkAgentRequest JSON example path.",
    )
    parser.add_argument(
        "--response",
        default=str(PROJECT_ROOT / "examples" / "topic11_response.json"),
        help="Expected ChunkAgentResponse JSON example path.",
    )
    parser.add_argument(
        "--skip-fastapi",
        action="store_true",
        help="Skip the FastAPI TestClient endpoint check.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)

    request_payload = _read_json(request_path)
    request = ChunkAgentRequest.model_validate(request_payload)
    print(f"OK request schema: {_display_path(request_path)}")

    actual_response = run_chunk_request(request)
    actual_payload = actual_response.model_dump(mode="json")

    expected_payload = _read_json(response_path)
    ChunkAgentResponse.model_validate(expected_payload)
    _assert_equal_json(expected_payload, actual_payload, response_path)
    print(f"OK service output matches: {_display_path(response_path)}")

    report = actual_payload.get("report", {})
    _assert_contract_report(report)
    print("OK report contract: GovernedDocument -> RagChunk[]")

    if not args.skip_fastapi:
        _check_fastapi_endpoint(request_payload, actual_payload)
        print("OK FastAPI endpoint: /api/v1/chunk")
        _check_langchain_endpoint(request_payload, actual_payload)
        print("OK LangChain endpoint: /api/v1/agent/run")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object.")
    return payload


def _assert_equal_json(expected: dict[str, Any], actual: dict[str, Any], response_path: Path) -> None:
    if expected == actual:
        return
    expected_text = _canonical_json(expected).splitlines()
    actual_text = _canonical_json(actual).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile=str(response_path),
            tofile="service output",
            lineterm="",
        )
    )
    raise AssertionError(f"Topic 11 response example is stale:\n{diff}")


def _assert_contract_report(report: dict[str, Any]) -> None:
    expected = {
        "schema_version": "hsc-agent-api-v1",
        "input_contract": "GovernedDocument",
        "output_contract": "RagChunk[]",
        "governance_stage": "post_normalization_packaging",
        "normalization_status": "provided_by_upstream",
    }
    missing_or_changed = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if missing_or_changed:
        raise AssertionError(f"Unexpected contract report fields: {missing_or_changed}")


def _check_fastapi_endpoint(request_payload: dict[str, Any], expected_payload: dict[str, Any]) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post("/api/v1/chunk", json=request_payload)
    if response.status_code != 200:
        raise AssertionError(f"FastAPI returned {response.status_code}: {response.text}")
    payload = response.json()
    if payload != expected_payload:
        raise AssertionError("FastAPI /api/v1/chunk response differs from service output.")


def _check_langchain_endpoint(request_payload: dict[str, Any], expected_payload: dict[str, Any]) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    agent_request = {
        "instruction": "Use LangChain tools to chunk this governed document for RAG.",
        "document": request_payload["document"],
        "strategy": request_payload.get("strategy", "hsc_rag"),
        "config": request_payload.get("config", {}),
        "include_report": request_payload.get("include_report", True),
        "llm_provider": "mock",
    }
    response = client.post("/api/v1/agent/run", json=agent_request)
    if response.status_code != 200:
        raise AssertionError(f"LangChain endpoint returned {response.status_code}: {response.text}")
    payload = response.json()
    if payload.get("provider") != "mock":
        raise AssertionError(f"Unexpected LangChain provider: {payload.get('provider')}")
    if payload.get("selected_tool") != "chunk_current_document":
        raise AssertionError(f"Unexpected LangChain tool: {payload.get('selected_tool')}")
    if payload.get("result") != expected_payload:
        raise AssertionError("LangChain endpoint result differs from deterministic chunk output.")


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
