# LangChain Agent Usage

This project keeps HSC-RAG chunking deterministic and auditable. LangChain is
used as the agent orchestration layer that exposes the existing HSC-RAG
capabilities as tools.

## What LangChain Does

- Wraps HSC-RAG chunking as `StructuredTool` tools.
- Provides an online agent endpoint: `POST /api/v1/agent/run`.
- Supports offline `mock` mode for reproducible demos without API keys.
- Supports optional `openai_compatible` mode through environment variables.

The default `mock` provider does not call an external model. It still invokes
the LangChain tool layer, then returns the deterministic HSC-RAG result.

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
