# Topic 11 JSON Contract

## Position In The Governance Pipeline

Topic 11 is a callable specialist agent in the data governance pipeline. It is
not responsible for raw PDF/Word parsing, data cleaning, terminology
normalization, or final result-package assembly.

Pipeline position:

```text
Topic 4 terminology normalization
-> normalized structured intermediate JSON
-> Topic 11 HSC-RAG chunking and content organization
-> chunk sequence JSON
-> Topic 5 format conversion and standard result package assembly
```

The formal service boundary is JSON request to JSON response. JSONL files in
`data/processed/**` are offline experiment artifacts for batch evaluation and
reproducibility, not the mandatory online integration format.

## Formal API

Single document:

```text
POST /api/v1/chunk
Content-Type: application/json
```

Batch documents:

```text
POST /api/v1/chunk/batch
Content-Type: application/json
```

LangChain-wrapped agent endpoint:

```text
POST /api/v1/agent/run
Content-Type: application/json
```

Recommended integration for Topic 5 is `/api/v1/chunk` or
`/api/v1/chunk/batch`. `/api/v1/agent/run` demonstrates LangChain tool
orchestration and can be used when the caller wants an agent-style wrapper.

## Input JSON

The input contract is `ChunkAgentRequest`.

```json
{
  "document": {
    "doc_id": "topic11_demo_doc",
    "dataset": "topic4_normalized_demo",
    "split": "handoff",
    "source_doc_id": "source_contract_001",
    "title": "Topic 11 JSON Handoff Demo",
    "normalization_status": "provided_by_upstream",
    "term_policy": "topic4_normalized_terms",
    "governance_stage": "post_normalization_packaging",
    "blocks": []
  },
  "strategy": "hsc_rag",
  "config": {
    "min_tokens": 180,
    "target_tokens": 512,
    "max_tokens": 900
  },
  "include_report": true
}
```

Full runnable request example:

```text
examples/topic11_request.json
```

### Required Document Fields

| Field | Type | Meaning |
|---|---|---|
| `doc_id` | string | Stable document id inside the governance pipeline. |
| `dataset` | string | Source corpus, project, or batch name. |
| `split` | string | Batch split or handoff stage name. |
| `source_doc_id` | string | Original upstream document id. |
| `title` | string | Document title after upstream governance. |
| `normalization_status` | enum | `provided_by_upstream`, `provided_by_dataset`, or `simulated_governed`. |
| `governance_stage` | string | Should usually be `post_normalization_packaging` for Topic 11. |
| `blocks` | array | Governed content blocks consumed by HSC-RAG. |

### Required Block Fields

| Field | Type | Meaning |
|---|---|---|
| `block_id` | string | Stable source block id. |
| `doc_id` | string | Parent document id. |
| `type` | enum | `paragraph`, `list`, `table`, `figure`, `code`, `formula`, etc. |
| `text` | string | Normalized text content. Topic 11 does not rewrite terminology. |
| `order` | integer | Original block order after upstream parsing/governance. |
| `level` | integer | Structural depth. |
| `title_path` | string[] | Heading path used for structure-aware chunking. |
| `source_anchor` | object | Source trace pointer used for original-text back links. |
| `entity_tags` | string[] | Standard entity tags from upstream normalization/linking when available. |
| `metadata` | object | Optional upstream metadata. |

## Output JSON

The output contract is `ChunkAgentResponse`.

```json
{
  "agent": "hsc-rag",
  "strategy": "hsc_rag",
  "doc_id": "topic11_demo_doc",
  "chunk_count": 1,
  "chunks": [],
  "report": {}
}
```

Full runnable response example:

```text
examples/topic11_response.json
```

Chinese teacher-facing handoff note:

```text
docs/topic11_handoff_contract_zh.md
```

### Required Chunk Fields

| Field | Type | Meaning |
|---|---|---|
| `chunk_id` | string | Stable chunk id. |
| `doc_id` | string | Parent document id. |
| `dataset` | string | Source corpus, project, or batch name. |
| `split` | string | Batch split or handoff stage name. |
| `strategy` | enum | Chunking strategy, usually `hsc_rag`. |
| `text` | string | Chunk text for downstream RAG indexing. |
| `token_count` | integer | Estimated token length. |
| `title_path` | string[] | Structural heading path retained for retrieval metadata. |
| `source_blocks` | string[] | Source block ids covered by the chunk. |
| `source_anchor` | object | Aggregated source range and assets. |
| `tags` | string[] | Topic/keyword tags for downstream organization. |
| `summary` | string | Faithful chunk summary. |
| `entity_tags` | string[] | Standard entity/entity-like tags. |
| `quality_flags` | string[] | Auditable quality labels such as `source_anchor_complete`. |
| `metadata` | object | Strategy parameters and optional enrichment metadata. |

## Topic 4 And Topic 5 Handoff Rules

- Topic 4 must output a governed structured JSON document whose terminology is
  already normalized.
- Topic 11 must not change normalized wording or perform field mapping.
- Topic 11 reads `title_path`, `block.type`, `source_anchor`, and
  `entity_tags` to produce traceable chunks.
- Topic 11 returns JSON chunks that Topic 5 can place into its standard result
  package.
- Topic 5 remains responsible for final package assembly, Markdown rendering,
  manifest/checksum generation, and target-schema field mapping.

## LangChain Usage

LangChain is used as the agent/tool orchestration layer. It wraps the local
deterministic HSC-RAG service as tools:

```text
inspect_hsc_rag_context
chunk_current_document
chunk_current_batch
```

The core chunking logic remains deterministic and auditable. This matches the
course requirement that data governance agents should be reproducible,
traceable, and service-callable.

## Validation Commands

Validate the formal request example, expected response example, FastAPI chunk
endpoint, and LangChain wrapper endpoint:

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_topic11_contract.py
```

Expected output:

```text
OK request schema: examples\topic11_request.json
OK service output matches: examples\topic11_response.json
OK report contract: GovernedDocument -> RagChunk[]
OK FastAPI endpoint: /api/v1/chunk
OK LangChain endpoint: /api/v1/agent/run
```

Run the formal JSON request through the service:

```powershell
$body = Get-Content examples\topic11_request.json -Raw
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/chunk `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Run the offline LangChain wrapper:

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_langchain_agent.py `
  --input data\processed\qasper\train\governed_documents.jsonl `
  --limit-docs 1 `
  --strategy hsc_rag `
  --provider mock
```
