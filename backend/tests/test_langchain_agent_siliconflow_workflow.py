from __future__ import annotations

from typing import Any

from app.core.schemas import LangChainAgentRequest
from app.services import langchain_agent_service
from app.services.langchain_agent_service import run_langchain_agent


class FakeRemoteJsonProvider:
    provider_name = "openai_compatible"

    def __init__(self, model: str) -> None:
        self.model = model

    def complete_json(self, messages, fallback):  # noqa: ANN001
        result = dict(fallback)
        result.update(
            {
                "summary": "Remote Qwen semantic organization summary.",
                "topic_tags": ["remote-qwen", "semantic-organization"],
                "entity_tags": ["HSC-RAG", "Qwen"],
                "semantic_integrity_score": 4.8,
                "summary_faithfulness_score": 4.7,
                "tag_accuracy_score": 4.6,
                "faithfulness_risk": "low",
                "quality_reason": "Remote model result is grounded in the chunk.",
                "qa_pairs": [],
                "provider_execution": "remote_llm_call",
            }
        )
        return result


def test_preferred_enrichment_tool_uses_openai_compatible_provider_without_router_call(
    monkeypatch,
    topic11_request_payload,
):
    captured: dict[str, Any] = {}

    def fake_build_json_provider(**kwargs):
        captured.update(kwargs)
        return FakeRemoteJsonProvider(model=kwargs["model"])

    monkeypatch.setattr(langchain_agent_service, "build_json_provider", fake_build_json_provider)

    request = LangChainAgentRequest(
        instruction=(
            "Use HSC-RAG to chunk this governed document, then use the LLM semantic "
            "organization skill to generate summaries, topic tags, entity tags, and scores."
        ),
        document=topic11_request_payload["document"],
        strategy="hsc_rag",
        config=topic11_request_payload["config"],
        include_report=True,
        preferred_tool="chunk_and_enrich_current_document",
        llm_provider="openai_compatible",
        llm_model="Qwen/Qwen3-VL-32B-Instruct",
        llm_base_url="https://api.siliconflow.cn/v1",
        llm_api_key_env="SILICONFLOW_API_KEY",
        llm_timeout_seconds=240,
        llm_use_response_format=False,
    )

    response = run_langchain_agent(request)

    assert response.selected_tool == "chunk_and_enrich_current_document"
    assert response.provider == "openai_compatible"
    assert response.warnings == []
    assert captured["provider"] == "openai_compatible"
    assert captured["model"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert captured["base_url"] == "https://api.siliconflow.cn/v1"
    assert captured["api_key_env"] == "SILICONFLOW_API_KEY"
    assert captured["timeout_seconds"] == 240
    assert captured["use_response_format"] is False

    report = response.result["report"]["llm_semantic_organization"]
    assert report["provider"] == "openai_compatible"
    assert report["model"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert report["provider_execution_counts"] == {"remote_llm_call": response.result["chunk_count"]}

    first_enrichment = response.result["chunks"][0]["metadata"]["llm_enrichment"]
    assert first_enrichment["provider_execution"] == "remote_llm_call"
    assert first_enrichment["summary"] == "Remote Qwen semantic organization summary."
