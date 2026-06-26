# -*- coding: utf-8 -*-
"""Provider adapters for the LLM semantic organization skill.

The chunking boundary itself is deterministic and auditable. This module only
adapts optional LLM calls used after chunking for summary/tag/entity enrichment
and quality judging. A deterministic mock provider is kept for offline replay.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class LlmMessage:
    role: str
    content: str


class JsonLlmProvider(Protocol):
    provider_name: str
    model: str

    def complete_json(self, messages: list[LlmMessage], fallback: JsonDict) -> JsonDict:
        """Return a JSON object produced by the provider."""


class MockJsonProvider:
    """Deterministic provider for demos, tests, and private/offline runs."""

    provider_name = "mock"

    def __init__(self, model: str = "mock-semantic-organizer-v1") -> None:
        self.model = model

    def complete_json(self, messages: list[LlmMessage], fallback: JsonDict) -> JsonDict:
        result = dict(fallback)
        result["provider_execution"] = "mock_offline_replay"
        result["provider_note"] = (
            "Deterministic fallback used for offline reproducibility. "
            "Switch provider to openai_compatible for real LLM calls."
        )
        return result


class OpenAICompatibleJsonProvider:
    """Minimal OpenAI-compatible chat/completions JSON provider.

    Works with OpenAI-compatible endpoints such as OpenAI, DeepSeek, Qwen,
    Zhipu-compatible gateways, vLLM, or Ollama OpenAI-compatible servers when
    their base URL and API key environment variable are supplied.
    """

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.1,
        max_tokens: int | None = 700,
        use_response_format: bool = True,
        timeout_seconds: float = 60.0,
        fallback_on_error: bool = True,
    ) -> None:
        self.base_url = _completion_url(base_url)
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_response_format = use_response_format
        self.timeout_seconds = timeout_seconds
        self.fallback_on_error = fallback_on_error

    def complete_json(self, messages: list[LlmMessage], fallback: JsonDict) -> JsonDict:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {self.api_key_env}. "
                "Use --provider mock for offline runs."
            )

        payload = {
            "model": self.model,
            "messages": [message.__dict__ for message in messages],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
            data = json.loads(body)
            content = data["choices"][0]["message"]["content"]
            result = extract_json_object(content)
            result["provider_execution"] = "remote_llm_call"
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            message = f"HTTP {exc.code}: {detail or exc.reason}"
            if not self.fallback_on_error:
                raise RuntimeError(f"LLM provider call failed: {message}") from exc
            result = dict(fallback)
            result["provider_execution"] = "fallback_after_provider_error"
            result["provider_error"] = message
            return result
        except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            if not self.fallback_on_error:
                raise RuntimeError(f"LLM provider call failed: {exc}") from exc
            result = dict(fallback)
            result["provider_execution"] = "fallback_after_provider_error"
            result["provider_error"] = str(exc)
            return result


def build_json_provider(
    *,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.1,
    max_tokens: int | None = 700,
    use_response_format: bool = True,
    timeout_seconds: float = 60.0,
    fallback_on_error: bool = True,
) -> JsonLlmProvider:
    provider = provider.strip().lower()
    if provider == "mock":
        return MockJsonProvider(model=model or "mock-semantic-organizer-v1")
    if provider == "openai_compatible":
        if not base_url:
            raise ValueError("--base-url is required for provider=openai_compatible")
        if not model:
            raise ValueError("--model is required for provider=openai_compatible")
        return OpenAICompatibleJsonProvider(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            use_response_format=use_response_format,
            timeout_seconds=timeout_seconds,
            fallback_on_error=fallback_on_error,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def extract_json_object(text: str) -> JsonDict:
    """Parse a JSON object from plain content or a fenced JSON block."""

    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)

    block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if block_match:
        return json.loads(block_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _completion_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"
