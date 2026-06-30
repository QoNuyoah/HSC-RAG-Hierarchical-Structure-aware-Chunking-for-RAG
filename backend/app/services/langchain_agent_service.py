# -*- coding: utf-8 -*-
"""LangChain agent wrapper for HSC-RAG tools.

The deterministic HSC-RAG chunkers remain the core method. LangChain is used as
the agent/tool orchestration layer so the project can expose the same chunking
ability through a standard agent interface.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from app.core.schemas import (
    ChunkAgentRequest,
    ChunkBatchAgentRequest,
    ChunkStrategy,
    GovernedDocument,
    LangChainAgentRequest,
    LangChainAgentResponse,
)
from app.llm.chunk_enricher import ChunkEnrichmentConfig, ChunkSemanticEnricher
from app.llm.providers import build_json_provider
from app.services.chunking_service import run_chunk_batch_request, run_chunk_request

try:
    import langchain
    from langchain_core.tools import StructuredTool
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    langchain = None
    StructuredTool = None  # type: ignore[assignment]
    _TOOL_IMPORT_ERROR: ImportError | None = exc
else:
    _TOOL_IMPORT_ERROR = None

try:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    create_agent = None  # type: ignore[assignment]
    ChatOpenAI = None  # type: ignore[assignment]
    _REMOTE_AGENT_IMPORT_ERROR: ImportError | None = exc
else:
    _REMOTE_AGENT_IMPORT_ERROR = None


SYSTEM_PROMPT = """You are the HSC-RAG LangChain agent.
You must use the available tools to operate on the governed document context
already attached to this request. Do not invent chunks. The chunk boundaries are
produced by deterministic HSC-RAG tools; your role is selecting and explaining
the right tool call. When the instruction asks for summaries, tags, entities,
semantic organization, or LLM participation, call the chunk-and-enrich tool so
the model contributes to post-chunk semantic organization instead of only
routing the request."""


class EmptyToolArgs(BaseModel):
    """No-argument tool input."""


class ChunkToolArgs(BaseModel):
    """Tool arguments for chunking the current request context."""

    strategy: ChunkStrategy | None = Field(default=None, description="Chunking strategy. Defaults to request.strategy.")
    config: dict[str, Any] | None = Field(default=None, description="Optional chunker config override.")
    include_report: bool | None = Field(default=None, description="Whether to include the chunk quality report.")


class _LangChainRuntime:
    def __init__(self, request: LangChainAgentRequest) -> None:
        self.request = request
        self.documents = _collect_documents(request)
        self.trace: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] = {}

    def build_tools(self) -> list[Any]:
        _ensure_tooling_available()

        def inspect_hsc_rag_context() -> str:
            """Inspect the current HSC-RAG request context and available tools."""

            return self._record_tool_result("inspect_hsc_rag_context", {}, self._context_summary())

        def chunk_current_document(
            strategy: ChunkStrategy | None = None,
            config: dict[str, Any] | None = None,
            include_report: bool | None = None,
        ) -> str:
            """Chunk the single governed document attached to the current request."""

            if len(self.documents) != 1:
                raise ValueError("chunk_current_document requires exactly one governed document in the request.")
            request = ChunkAgentRequest(
                document=self.documents[0],
                strategy=strategy or self.request.strategy,
                config=self._resolve_config(config),
                include_report=self._resolve_include_report(include_report),
            )
            response = run_chunk_request(request).model_dump(mode="json")
            return self._record_tool_result(
                "chunk_current_document",
                {
                    "strategy": request.strategy,
                    "config": request.config,
                    "include_report": request.include_report,
                },
                response,
            )

        def chunk_current_batch(
            strategy: ChunkStrategy | None = None,
            config: dict[str, Any] | None = None,
            include_report: bool | None = None,
        ) -> str:
            """Chunk every governed document attached to the current request."""

            if not self.documents:
                raise ValueError("chunk_current_batch requires at least one governed document in the request.")
            request = ChunkBatchAgentRequest(
                documents=self.documents,
                strategy=strategy or self.request.strategy,
                config=self._resolve_config(config),
                include_report=self._resolve_include_report(include_report),
            )
            response = run_chunk_batch_request(request).model_dump(mode="json")
            return self._record_tool_result(
                "chunk_current_batch",
                {
                    "strategy": request.strategy,
                    "config": request.config,
                    "include_report": request.include_report,
                    "document_count": len(self.documents),
                },
                response,
            )

        def chunk_and_enrich_current_document(
            strategy: ChunkStrategy | None = None,
            config: dict[str, Any] | None = None,
            include_report: bool | None = None,
        ) -> str:
            """Chunk the current document and run LLM semantic organization over the chunks."""

            if len(self.documents) != 1:
                raise ValueError("chunk_and_enrich_current_document requires exactly one governed document.")
            request = ChunkAgentRequest(
                document=self.documents[0],
                strategy=strategy or self.request.strategy,
                config=self._resolve_config(config),
                include_report=self._resolve_include_report(include_report),
            )
            response = run_chunk_request(request)
            enriched_chunks = self._enrich_chunks(response.chunks)
            response = response.model_copy(update={"chunks": enriched_chunks})
            payload = response.model_dump(mode="json")
            if request.include_report:
                payload.setdefault("report", {})["llm_semantic_organization"] = self._llm_report(enriched_chunks)
            return self._record_tool_result(
                "chunk_and_enrich_current_document",
                {
                    "strategy": request.strategy,
                    "config": request.config,
                    "include_report": request.include_report,
                    "llm_provider": self.request.llm_provider,
                    "llm_model": self.request.llm_model or self._default_enrichment_model(),
                },
                payload,
            )

        def chunk_and_enrich_current_batch(
            strategy: ChunkStrategy | None = None,
            config: dict[str, Any] | None = None,
            include_report: bool | None = None,
        ) -> str:
            """Chunk every current document and run LLM semantic organization over all chunks."""

            if not self.documents:
                raise ValueError("chunk_and_enrich_current_batch requires at least one governed document.")
            request = ChunkBatchAgentRequest(
                documents=self.documents,
                strategy=strategy or self.request.strategy,
                config=self._resolve_config(config),
                include_report=self._resolve_include_report(include_report),
            )
            response = run_chunk_batch_request(request)
            enriched_results = []
            for item in response.results:
                enriched_chunks = self._enrich_chunks(item.chunks)
                result_payload = item.model_copy(update={"chunks": enriched_chunks}).model_dump(mode="json")
                if request.include_report:
                    result_payload.setdefault("report", {})["llm_semantic_organization"] = self._llm_report(
                        enriched_chunks
                    )
                enriched_results.append(result_payload)
            payload = response.model_dump(mode="json")
            payload["results"] = enriched_results
            return self._record_tool_result(
                "chunk_and_enrich_current_batch",
                {
                    "strategy": request.strategy,
                    "config": request.config,
                    "include_report": request.include_report,
                    "document_count": len(self.documents),
                    "llm_provider": self.request.llm_provider,
                    "llm_model": self.request.llm_model or self._default_enrichment_model(),
                },
                payload,
            )

        return [
            StructuredTool.from_function(
                inspect_hsc_rag_context,
                name="inspect_hsc_rag_context",
                description="Inspect the attached governed documents and explain available HSC-RAG tools.",
                args_schema=EmptyToolArgs,
            ),
            StructuredTool.from_function(
                chunk_current_document,
                name="chunk_current_document",
                description="Run deterministic HSC-RAG chunking for the single attached governed document.",
                args_schema=ChunkToolArgs,
            ),
            StructuredTool.from_function(
                chunk_current_batch,
                name="chunk_current_batch",
                description="Run deterministic HSC-RAG chunking for all attached governed documents.",
                args_schema=ChunkToolArgs,
            ),
            StructuredTool.from_function(
                chunk_and_enrich_current_document,
                name="chunk_and_enrich_current_document",
                description=(
                    "Run HSC-RAG chunking for one governed document, then use the configured LLM "
                    "semantic organization skill to add faithful summaries, topic tags, entity tags, "
                    "semantic integrity scores, and quality reasons."
                ),
                args_schema=ChunkToolArgs,
            ),
            StructuredTool.from_function(
                chunk_and_enrich_current_batch,
                name="chunk_and_enrich_current_batch",
                description=(
                    "Run HSC-RAG chunking for all governed documents, then enrich every chunk with "
                    "LLM semantic organization metadata."
                ),
                args_schema=ChunkToolArgs,
            ),
        ]

    def invoke_tool(self, tool_name: str, tools: list[Any]) -> None:
        tool = next((candidate for candidate in tools if candidate.name == tool_name), None)
        if tool is None:
            raise HTTPException(status_code=400, detail=f"Unknown LangChain tool: {tool_name}")
        try:
            if tool_name == "inspect_hsc_rag_context":
                tool.invoke({})
                return
            tool.invoke(
                {
                    "strategy": self.request.strategy,
                    "config": self.request.config,
                    "include_report": self.request.include_report,
                }
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"LangChain tool {tool_name} failed: {exc}") from exc

    def _context_summary(self) -> dict[str, Any]:
        return {
            "agent": "hsc-rag-langchain",
            "document_count": len(self.documents),
            "strategy": self.request.strategy,
            "config": self.request.config,
            "preferred_tool": self.request.preferred_tool,
            "available_tools": [
                "inspect_hsc_rag_context",
                "chunk_current_document",
                "chunk_current_batch",
                "chunk_and_enrich_current_document",
                "chunk_and_enrich_current_batch",
            ],
            "core_method": "deterministic hierarchical structure-aware chunking",
            "llm_semantic_role": "post-chunk summary, topic tags, entity tags, semantic integrity scoring",
            "langchain_role": "tool orchestration, optional LLM semantic organization, and agent interface",
        }

    def _record_tool_result(self, tool_name: str, args: dict[str, Any], payload: dict[str, Any]) -> str:
        compact = _compact_result(payload)
        self.last_result = payload
        self.trace.append(
            {
                "tool": tool_name,
                "args": args,
                "output_summary": compact,
            }
        )
        return json.dumps(compact, ensure_ascii=False)

    def _resolve_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        if config:
            return dict(config)
        return dict(self.request.config)

    def _resolve_include_report(self, include_report: bool | None) -> bool:
        if include_report is None:
            return self.request.include_report
        return include_report

    def _enrich_chunks(self, chunks):
        enricher = ChunkSemanticEnricher(
            provider=self._build_enrichment_provider(),
            config=ChunkEnrichmentConfig(include_qa=_wants_qa_generation(self.request.instruction)),
        )
        return [enricher.enrich_chunk(chunk) for chunk in chunks]

    def _build_enrichment_provider(self):
        base_url = self.request.llm_base_url
        if self.request.llm_provider == "openai_compatible" and not base_url:
            base_url = "https://api.openai.com/v1"
        return build_json_provider(
            provider=self.request.llm_provider,
            model=self.request.llm_model or self._default_enrichment_model(),
            base_url=base_url,
            api_key_env=self.request.llm_api_key_env,
            temperature=self.request.llm_temperature,
            timeout_seconds=self.request.llm_timeout_seconds,
            use_response_format=self.request.llm_use_response_format,
            fallback_on_error=True,
        )

    def _default_enrichment_model(self) -> str:
        if self.request.llm_provider == "mock":
            return "mock-semantic-organizer-v1"
        return "gpt-4.1-mini"

    def _llm_report(self, chunks) -> dict[str, Any]:
        enrichments = [
            (chunk.metadata or {}).get("llm_enrichment", {})
            for chunk in chunks
            if isinstance((chunk.metadata or {}).get("llm_enrichment"), dict)
        ]
        if not enrichments:
            return {"chunks": 0}

        def avg(key: str) -> float:
            values = [float(item[key]) for item in enrichments if isinstance(item.get(key), (int, float))]
            return round(sum(values) / len(values), 4) if values else 0.0

        risk_counts: dict[str, int] = {}
        provider_execution_counts: dict[str, int] = {}
        for item in enrichments:
            risk = str(item.get("faithfulness_risk", "unknown"))
            provider_execution = str(item.get("provider_execution", "unknown"))
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            provider_execution_counts[provider_execution] = provider_execution_counts.get(provider_execution, 0) + 1
        return {
            "skill": "llm_semantic_organization",
            "chunks": len(enrichments),
            "provider": self.request.llm_provider,
            "model": self.request.llm_model or self._default_enrichment_model(),
            "semantic_integrity_score_avg": avg("semantic_integrity_score"),
            "summary_faithfulness_score_avg": avg("summary_faithfulness_score"),
            "tag_accuracy_score_avg": avg("tag_accuracy_score"),
            "faithfulness_risk_counts": dict(sorted(risk_counts.items())),
            "provider_execution_counts": dict(sorted(provider_execution_counts.items())),
            "prompt_version": enrichments[0].get("prompt_version"),
        }


def run_langchain_agent(request: LangChainAgentRequest) -> LangChainAgentResponse:
    """Run the HSC-RAG LangChain agent with offline mock or remote LLM routing."""

    runtime = _LangChainRuntime(request)
    tools = runtime.build_tools()
    warnings: list[str] = []
    selected_tool: str | None = None
    answer = ""

    if request.preferred_tool:
        selected_tool = _select_tool(request, len(runtime.documents))
        runtime.invoke_tool(selected_tool, tools)
        answer = _tool_answer(selected_tool, runtime.last_result)
        if request.llm_provider == "mock":
            warnings.append("mock provider used: LangChain tools were invoked without an external LLM call.")
    elif request.llm_provider == "mock":
        selected_tool = _select_tool(request, len(runtime.documents))
        runtime.invoke_tool(selected_tool, tools)
        answer = _tool_answer(selected_tool, runtime.last_result)
        warnings.append("mock provider used: LangChain tools were invoked without an external LLM call.")
    else:
        selected_tool, answer = _run_remote_langchain_agent(request, runtime, tools, warnings)
        if not runtime.last_result:
            fallback_tool = _select_tool(request, len(runtime.documents))
            runtime.invoke_tool(fallback_tool, tools)
            selected_tool = fallback_tool
            warnings.append("remote agent did not call a tool; deterministic fallback tool was executed.")

    return LangChainAgentResponse(
        provider=request.llm_provider,
        model=request.llm_model or ("mock-langchain-router-v1" if request.llm_provider == "mock" else None),
        langchain_version=getattr(langchain, "__version__", None) if langchain is not None else None,
        instruction=request.instruction,
        selected_tool=selected_tool,
        answer=answer or _tool_answer(selected_tool or "inspect_hsc_rag_context", runtime.last_result),
        tool_trace=runtime.trace,
        result=runtime.last_result,
        warnings=warnings,
    )


def _run_remote_langchain_agent(
    request: LangChainAgentRequest,
    runtime: _LangChainRuntime,
    tools: list[Any],
    warnings: list[str],
) -> tuple[str | None, str]:
    _ensure_remote_agent_available()
    if not request.llm_model:
        raise HTTPException(status_code=400, detail="llm_model is required when llm_provider=openai_compatible.")
    api_key = os.getenv(request.llm_api_key_env)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"Missing API key environment variable: {request.llm_api_key_env}. Use llm_provider=mock offline.",
        )

    model_kwargs: dict[str, Any] = {
        "model": request.llm_model,
        "api_key": api_key,
        "temperature": request.llm_temperature,
        "timeout": request.llm_timeout_seconds,
    }
    if request.llm_base_url:
        model_kwargs["base_url"] = request.llm_base_url

    llm = ChatOpenAI(**model_kwargs)
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    try:
        state = agent.invoke({"messages": [{"role": "user", "content": _agent_user_message(request, runtime)}]})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LangChain remote agent failed: {exc}") from exc
    answer = _last_message_content(state)
    selected_tool = runtime.trace[-1]["tool"] if runtime.trace else None
    if not selected_tool:
        warnings.append("remote agent returned without a recorded LangChain tool call.")
    return selected_tool, answer


def _agent_user_message(request: LangChainAgentRequest, runtime: _LangChainRuntime) -> str:
    summary = runtime._context_summary()
    return (
        f"Instruction: {request.instruction}\n"
        f"Request context JSON: {json.dumps(summary, ensure_ascii=False)}\n"
        "Call the most appropriate tool exactly once, then summarize the result."
    )


def _last_message_content(state: Any) -> str:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", None)
    if content is None and isinstance(messages[-1], dict):
        content = messages[-1].get("content")
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content or "")


def _collect_documents(request: LangChainAgentRequest) -> list[GovernedDocument]:
    documents: list[GovernedDocument] = []
    if request.document is not None:
        documents.append(request.document)
    documents.extend(request.documents)
    return documents


def _select_tool(request: LangChainAgentRequest, document_count: int) -> str:
    allowed = {
        "inspect_hsc_rag_context",
        "chunk_current_document",
        "chunk_current_batch",
        "chunk_and_enrich_current_document",
        "chunk_and_enrich_current_batch",
    }
    if request.preferred_tool:
        if request.preferred_tool not in allowed:
            raise HTTPException(
                status_code=400,
                detail={"message": "Unknown preferred_tool", "allowed_tools": sorted(allowed)},
            )
        return request.preferred_tool
    if document_count == 0:
        return "inspect_hsc_rag_context"
    if _wants_llm_enrichment(request.instruction):
        if document_count == 1:
            return "chunk_and_enrich_current_document"
        return "chunk_and_enrich_current_batch"
    if document_count == 1:
        return "chunk_current_document"
    return "chunk_current_batch"


def _tool_answer(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name == "inspect_hsc_rag_context":
        return "LangChain agent inspected the HSC-RAG context and available tools."
    if tool_name == "chunk_current_document":
        return (
            "LangChain agent invoked the HSC-RAG chunking tool for one governed document "
            f"and produced {result.get('chunk_count', 0)} chunks."
        )
    if tool_name == "chunk_and_enrich_current_document":
        llm_report = result.get("report", {}).get("llm_semantic_organization", {})
        return (
            "LangChain agent invoked HSC-RAG chunking and the LLM semantic organization "
            f"tool for one governed document, producing {result.get('chunk_count', 0)} chunks "
            f"with {llm_report.get('chunks', 0)} enriched chunks."
        )
    if tool_name == "chunk_and_enrich_current_batch":
        return (
            "LangChain agent invoked HSC-RAG batch chunking and LLM semantic organization "
            f"for {result.get('document_count', 0)} documents and produced "
            f"{result.get('total_chunks', 0)} chunks."
        )
    return (
        "LangChain agent invoked the HSC-RAG batch chunking tool "
        f"for {result.get('document_count', 0)} documents and produced {result.get('total_chunks', 0)} chunks."
    )


def _compact_result(payload: dict[str, Any]) -> dict[str, Any]:
    if "chunks" in payload:
        return {
            "agent": payload.get("agent"),
            "strategy": payload.get("strategy"),
            "doc_id": payload.get("doc_id"),
            "chunk_count": payload.get("chunk_count"),
            "report": payload.get("report", {}),
            "chunk_preview": [_chunk_preview(chunk) for chunk in payload.get("chunks", [])[:3]],
        }
    if "results" in payload:
        return {
            "agent": payload.get("agent"),
            "strategy": payload.get("strategy"),
            "document_count": payload.get("document_count"),
            "total_chunks": payload.get("total_chunks"),
            "results": [
                {
                    "doc_id": item.get("doc_id"),
                    "chunk_count": item.get("chunk_count"),
                    "report": item.get("report", {}),
                    "chunk_preview": [_chunk_preview(chunk) for chunk in item.get("chunks", [])[:2]],
                }
                for item in payload.get("results", [])[:5]
            ],
        }
    return payload


def _chunk_preview(chunk: dict[str, Any]) -> dict[str, Any]:
    enrichment = (chunk.get("metadata") or {}).get("llm_enrichment") or {}
    return {
        "chunk_id": chunk.get("chunk_id"),
        "token_count": chunk.get("token_count"),
        "title_path": chunk.get("title_path", []),
        "source_blocks": chunk.get("source_blocks", [])[:5],
        "quality_flags": chunk.get("quality_flags", []),
        "summary": chunk.get("summary"),
        "boundary_decision": (chunk.get("metadata") or {}).get("closing_boundary_decision"),
        "llm_enrichment": {
            "summary": enrichment.get("summary"),
            "topic_tags": enrichment.get("topic_tags", [])[:4],
            "semantic_integrity_score": enrichment.get("semantic_integrity_score"),
            "summary_faithfulness_score": enrichment.get("summary_faithfulness_score"),
            "provider_execution": enrichment.get("provider_execution"),
        }
        if enrichment
        else None,
    }


def _wants_llm_enrichment(instruction: str) -> bool:
    text = instruction.lower()
    keywords = [
        "llm",
        "large model",
        "semantic organization",
        "semantic enrichment",
        "summary",
        "summarize",
        "tag",
        "entity",
        "qa",
        "大模型",
        "语义组织",
        "语义增强",
        "摘要",
        "标签",
        "实体",
        "问答",
    ]
    return any(keyword in text for keyword in keywords)


def _wants_qa_generation(instruction: str) -> bool:
    text = instruction.lower()
    return any(keyword in text for keyword in ["qa", "question", "instruction data", "问答", "指令数据"])


def _ensure_tooling_available() -> None:
    if _TOOL_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"LangChain tooling is not installed: {_TOOL_IMPORT_ERROR}",
        )


def _ensure_remote_agent_available() -> None:
    if _REMOTE_AGENT_IMPORT_ERROR is not None:
        raise HTTPException(
            status_code=500,
            detail=f"LangChain remote agent dependencies are not installed: {_REMOTE_AGENT_IMPORT_ERROR}",
        )
