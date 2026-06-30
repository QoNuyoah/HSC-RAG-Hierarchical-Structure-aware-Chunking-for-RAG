# LangChain Agent Usage

This project keeps HSC-RAG chunking deterministic and auditable. LangChain is
used as the agent orchestration layer that exposes the existing HSC-RAG
capabilities as tools.

## What LangChain Does

- Wraps HSC-RAG chunking as `StructuredTool` tools.
- Provides `chunk_and_enrich_current_document`, which first runs HSC-RAG
  chunking and then calls the LLM semantic organization skill for summaries,
  topic tags, entity tags, semantic integrity scores, and faithfulness signals.
- Provides an online agent endpoint: `POST /api/v1/agent/run`.
- Supports offline `mock` mode for reproducible demos without API keys.
- Supports optional `openai_compatible` mode through environment variables.

The default `mock` provider does not call an external model. It still invokes
the LangChain tool layer, then returns the deterministic HSC-RAG result.
When the instruction asks for LLM enrichment, summaries, tags, entities, or
semantic organization, the mock provider uses the same tool path but writes a
deterministic `metadata.llm_enrichment` record for offline replay.

## Run From CLI

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_langchain_agent.py `
  --input data\processed\qasper\train\governed_documents.jsonl `
  --limit-docs 1 `
  --strategy hsc_rag `
  --provider mock `
  --output runs\langchain_agent_demo.json
```

## Run The API

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe -m uvicorn app.main:app `
  --app-dir backend `
  --host 127.0.0.1 `
  --port 8000
```

Main endpoint:

```text
POST http://127.0.0.1:8000/api/v1/agent/run
```

Minimal request shape:

```json
{
  "instruction": "Use LangChain tools to chunk this governed document for RAG.",
  "documents": [
    {
      "doc_id": "demo_doc",
      "dataset": "demo",
      "split": "dev",
      "source_doc_id": "demo_doc",
      "title": "Demo",
      "normalization_status": "simulated_governed",
      "blocks": [
        {
          "block_id": "demo_b1",
          "doc_id": "demo_doc",
          "type": "paragraph",
          "text": "This is governed text ready for HSC-RAG chunking.",
          "order": 1,
          "title_path": ["Demo"],
          "source_anchor": {
            "dataset": "demo",
            "split": "dev",
            "source_doc_id": "demo_doc"
          }
        }
      ]
    }
  ],
  "strategy": "hsc_rag",
  "llm_provider": "mock"
}
```

## OpenAI-Compatible Mode

Do not put API keys in request JSON or source code. Store them in an
environment variable and pass only the variable name.

```powershell
$env:DEEPSEEK_API_KEY="your key"
```

Request fields:

```json
{
  "llm_provider": "openai_compatible",
  "llm_model": "deepseek-chat",
  "llm_base_url": "https://api.deepseek.com/v1",
  "llm_api_key_env": "DEEPSEEK_API_KEY"
}
```

The remote model selects a LangChain tool. The tool still calls the local
deterministic HSC-RAG implementation.

## LLM Semantic Organization Tool

To demonstrate that the large model is not just a routing shell, ask the agent
for semantic organization explicitly:

```json
{
  "instruction": "Use HSC-RAG to chunk this governed document, then use the LLM semantic organization skill to generate faithful summaries, topic tags, entity tags, and semantic integrity scores.",
  "document": {},
  "strategy": "hsc_rag",
  "llm_provider": "mock"
}
```

Expected `tool_trace`:

```text
chunk_and_enrich_current_document
```

The returned chunks contain:

```text
metadata.boundary_policy
metadata.closing_boundary_decision
metadata.llm_enrichment.summary
metadata.llm_enrichment.topic_tags
metadata.llm_enrichment.entity_tags
metadata.llm_enrichment.semantic_integrity_score
metadata.llm_enrichment.summary_faithfulness_score
```

In `openai_compatible` mode, the same tool calls the configured remote or local
OpenAI-compatible model for the semantic organization step.

## SiliconFlow Qwen API Demo

For SiliconFlow Qwen models, use `preferred_tool` to make the workflow explicit:
HSC-RAG first creates deterministic chunks, then Qwen is called only for the
semantic organization step. This avoids relying on remote tool-calling support
for tool selection.

Start the backend:

```powershell
cd /d E:\practical_training\HSC_RAG
& E:\anaconda3\envs\HSC_RAG\python.exe -m uvicorn app.main:app `
  --app-dir backend `
  --host 127.0.0.1 `
  --port 8000
```

Call the endpoint:

```powershell
cd /d E:\practical_training\HSC_RAG
$env:SILICONFLOW_API_KEY = "your SiliconFlow API key"

$request = Get-Content .\examples\topic11_request.json -Raw -Encoding UTF8 | ConvertFrom-Json
$body = @{
  instruction = "Use HSC-RAG to chunk this governed document, then use the LLM semantic organization skill to generate faithful summaries, topic tags, entity tags, and semantic integrity scores."
  document = $request.document
  strategy = "hsc_rag"
  config = $request.config
  include_report = $true
  preferred_tool = "chunk_and_enrich_current_document"
  llm_provider = "openai_compatible"
  llm_model = "Qwen/Qwen3-VL-32B-Instruct"
  llm_base_url = "https://api.siliconflow.cn/v1"
  llm_api_key_env = "SILICONFLOW_API_KEY"
  llm_temperature = 0.1
  llm_timeout_seconds = 240
  llm_use_response_format = $false
} | ConvertTo-Json -Depth 80

$response = Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/agent/run `
  -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body $body

$response.selected_tool
$response.result.report.llm_semantic_organization
$response.result.chunks[0].metadata.llm_enrichment
```

Evidence of a real model call:

```text
selected_tool = chunk_and_enrich_current_document
provider = openai_compatible
model = Qwen/Qwen3-VL-32B-Instruct
provider_execution_counts.remote_llm_call > 0
metadata.llm_enrichment.provider_execution = remote_llm_call
```

If `provider_execution` is `fallback_after_provider_error`, the request reached
the enrichment provider path but the remote model call failed or returned an
unsupported response. Check the API key, model name, network, timeout, and
`llm_use_response_format = false`.
