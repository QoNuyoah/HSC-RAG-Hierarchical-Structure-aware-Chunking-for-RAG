# HSC-RAG: Hierarchical Structure-aware Chunking for RAG

面向 RAG 的层级结构感知分段与内容组织智能体。

本项目对应课题 11“面向 RAG 的智能分段与内容组织智能体”。项目定位不是通用 PDF/Word 解析器，也不是上游数据清洗系统，而是数据治理流水线中的成果封装环节：

> 在数据治理中，分段发生在口径已统一之后、成果封装之时，以保证每个片段（chunk）内部术语口径一致、可直接供下游消费。

因此，HSC-RAG 的核心输入是 `GovernedDocument`，即已经由公开数据集或上游治理流程提供的结构化、可追溯、口径一致的文档对象。HSC-RAG 负责将其组织为适合 RAG 检索和生成消费的 chunks，并通过公开数据集的 question/evidence 标注进行自动评估。

## 当前完成状态

本仓库已经完成一条可运行的端到端实验链路：

```text
QASPER -> GovernedDocument
-> fixed / recursive / semantic / hsc_rag 四种分段策略
-> BM25 / Dense FAISS / Hybrid 三类检索器
-> Recall@1/3/5、MRR、nDCG@5 自动评估
-> FastAPI 后端 API
-> React 前端实验看板和 bad case 对比页面
```

已实现内容：

- 可运行的 HSC-RAG 分段智能体后端服务，提供标准 HTTP 接口供上游转换流水线调用。
- 标准在线接口：输入 `GovernedDocument` 结构化全文，输出 `RagChunk[]` chunk 序列。
- `QASPER -> GovernedDocument` 数据适配器。
- `GovernedDocument / GovernedBlock / GovernedQuery / RagChunk` 等核心数据契约。
- 四种分段策略：
  - `fixed`：固定窗口分段基线。
  - `recursive`：通用递归切分基线。
  - `semantic`：基于句间 TF-IDF 语义距离的语义切分基线。
  - `hsc_rag`：层级结构感知分段方法。
- 三种检索器：
  - `bm25`
  - `dense`：TF-IDF + SVD + FAISS，本地可复现，不依赖联网模型下载。
  - `hybrid`：BM25 + Dense score fusion。
- 自动评估脚本：
  - `Recall@1`
  - `Recall@3`
  - `Recall@5`
  - `MRR`
  - `nDCG@5`
- FastAPI 实验结果 API。
- React 前端实验看板。
- 同一 query 下 fixed / recursive / semantic / HSC-RAG 的 Top-5 bad case 对比页面。

## 项目结构

```text
HSC_RAG
├── backend
│   └── app
│       ├── adapters
│       │   └── qasper_adapter.py
│       ├── chunkers
│       │   ├── common.py
│       │   ├── fixed.py
│       │   ├── recursive.py
│       │   ├── semantic.py
│       │   └── hsc_rag.py
│       ├── core
│       │   └── schemas.py
│       ├── retrievers
│       │   ├── bm25.py
│       │   ├── dense_faiss.py
│       │   └── hybrid.py
│       ├── services
│       │   └── evaluation_store.py
│       └── main.py
├── data
│   └── processed
│       └── qasper
│           └── train
├── docs
├── frontend
│   ├── src
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── main.tsx
│   │   └── styles.css
│   └── package.json
├── reports
└── scripts
    ├── convert_qasper.py
    ├── convert_dureader.py
    ├── convert_hotpotqa.py
    ├── run_agent_pipeline.py
    ├── run_langchain_agent.py
    ├── run_chunking.py
    ├── run_llm_enrichment.py
    ├── run_retrieval_eval.py
    ├── validate_chunks.py
    └── validate_governed_outputs.py
```

## 环境要求

推荐环境：

```text
Python 3.11
Node.js 25.x
npm 11.x
```

当前开发环境：

```text
Conda env: HSC_RAG
Python: E:\anaconda3\envs\HSC_RAG\python.exe
```

主要 Python 依赖：

```text
fastapi
uvicorn
pydantic
rank-bm25
faiss-cpu
scikit-learn
numpy
pandas
sentence-transformers
langchain
langchain-openai
langgraph
```

主要前端依赖：

```text
React
TypeScript
Vite
lucide-react
```

## 数据说明

本项目使用公开数据集 QASPER 作为主要实验数据来源。

本仓库保留了小规模处理后的实验产物：

```text
data\processed\qasper\train
```

另外，本项目补充跑通了中文公开数据集 DuReader，用于验证 HSC-RAG 在中文网页问答场景下的跨数据集适配能力。DuReader 原始压缩包较大，未纳入 Git 版本管理；本仓库保留 50 条 `search.dev` 样本的标准化处理产物：

```text
data\processed\dureader\search_dev
```

此外，本项目补充适配了 HotpotQA 作为进阶多跳问答实验。HotpotQA 的 `supporting_facts` 可直接映射为 `gold_block_ids`，适合观察跨候选文章证据组织、Top-5 证据覆盖和 bad case 边界。它不替代 QASPER 主实验，因为 fixed chunk 在 HotpotQA same-doc 评估中容易通过跨文章混合获得较高排序指标。

```text
data\processed\hotpotqa\train_50
reports\hotpotqa_eval_summary.md
```

## 数据转换

在项目根目录执行：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\convert_qasper.py --split train --limit-docs 5
```

输出目录：

```text
data\processed\qasper\train
```

输出文件：

```text
governed_documents.jsonl
blocks.jsonl
queries.csv
gold_evidence.jsonl
conversion_report.json
```

验证转换结果：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_governed_outputs.py data\processed\qasper\train
```

当前转换结果：

```text
documents: 5
blocks: 420
queries: 33
answerable_queries: 28
unanswerable_queries: 5
evidence_items: 94
matched_evidence_items: 93
evidence_match_rate: 0.9894
```

DuReader 转换示例：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\convert_dureader.py --zip DuReader.zip --source search --split dev --limit-docs 50 --output-dir data\processed\dureader\search_dev
```

当前 DuReader 转换结果：

```text
documents: 50
blocks: 1870
queries: 50
answerable_queries: 49
gold_evidence_items: 91
evidence_match_rate: 1.0
```

HotpotQA 转换示例：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\convert_hotpotqa.py --zip HotpotQA.zip --split train --limit-docs 50 --output-dir data\processed\hotpotqa\train_50
```

当前 HotpotQA 转换结果：

```text
documents: 50
blocks: 2066
queries: 50
answerable_queries: 50
gold_evidence_items: 130
evidence_match_rate: 1.0
```

## 运行分段实验

生成四类 chunk：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_chunking.py --input data\processed\qasper\train\governed_documents.jsonl --output-dir data\processed\qasper\train --strategies fixed,recursive,semantic,hsc_rag
```

验证 HSC-RAG chunks：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_chunks.py data\processed\qasper\train\chunks_hsc_rag.jsonl --blocks data\processed\qasper\train\blocks.jsonl --max-tokens 900
```

当前 chunk 结构结果：

| Strategy | Chunks | Avg Tokens | Max Tokens | Title Consistent | Mixed Title Paths |
|---|---:|---:|---:|---:|---:|
| fixed | 50 | 446.28 | 511 | 7 | 32 |
| recursive | 52 | 441.98 | 510 | 7 | 36 |
| semantic | 101 | 220.93 | 501 | 45 | 38 |
| hsc_rag | 61 | 386.85 | 900 | 38 | 16 |

结构观察：

- fixed 和 recursive 的长度较稳定，但跨章节混合较严重。
- semantic 的 chunk 更细，title 一致性较高，但 chunk 数量明显增多，证据容易被拆散。
- HSC-RAG 在保持适中 chunk 数量的同时，将 `mixed_title_paths` 从 fixed 的 32 降到 16，更符合数据治理场景下“结构一致、可追溯、可消费”的要求。

HSC-RAG 主算法会为每个收束边界记录可解释评分，写入 `metadata.closing_boundary_decision`：

```text
boundary_score = structure_signal * w_structure
               + semantic_distance * w_semantic
               + length_pressure * w_length
```

其中 `structure_signal` 来自标题路径、顶层章节、保护块切换等结构证据；`semantic_distance` 来自当前 buffer 与右侧候选块的本地词袋余弦距离；`length_pressure` 表示当前 chunk 接近目标长度的程度。`split_reason` 会记录为 `semantic_boundary`、`section_boundary_respected`、`target_length_boundary` 或 `max_length_boundary`，用于证明主算法是“结构/语义感知分段”，而不是普通定长或纯标题规则切分。

## 运行检索评估

运行 BM25 / Dense / Hybrid 三类检索评估：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_retrieval_eval.py --chunk-dir data\processed\qasper\train --gold-evidence data\processed\qasper\train\gold_evidence.jsonl --strategies fixed,recursive,semantic,hsc_rag --retrievers bm25,dense,hybrid --top-k 1,3,5 --ndcg-k 5 --dense-encoder tfidf_svd --dense-svd-dim 128 --hybrid-alpha 0.55
```

主实验输出：

```text
data\processed\qasper\train\retrieval_eval_multi_summary.json
```

逐问题结果示例：

```text
retrieval_results_hsc_rag_bm25.jsonl
retrieval_results_hsc_rag_dense.jsonl
retrieval_results_hsc_rag_hybrid.jsonl
```

## 主实验结果

实验口径：

- 数据：QASPER train 子集，5 篇论文。
- 查询：33 个问题，其中 28 个 answerable query 参与评估。
- 检索范围：same-doc evidence retrieval。
- 评价依据：chunk 的 `source_blocks` 是否覆盖 `gold_block_ids`。
- 索引字段：主实验仅使用 `chunk.text`。

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | BM25 | 0.157738 | 0.437500 | 0.633929 | 0.450993 | 0.438077 |
| recursive | BM25 | 0.193452 | 0.437500 | 0.741071 | 0.477737 | 0.481412 |
| semantic | BM25 | 0.202381 | 0.437500 | 0.562500 | 0.478301 | 0.377404 |
| hsc_rag | BM25 | 0.217262 | 0.568452 | 0.723214 | 0.515704 | 0.509134 |
| hsc_rag | Dense FAISS | 0.258929 | 0.532738 | 0.651786 | 0.509752 | 0.489759 |
| hsc_rag | Hybrid | 0.288690 | 0.514881 | 0.705357 | 0.550271 | 0.526316 |

在 BM25 检索下，HSC-RAG 相对 fixed 的提升：

| Metric | Delta |
|---|---:|
| Recall@1 | +0.059524 |
| Recall@3 | +0.130952 |
| Recall@5 | +0.089285 |
| MRR | +0.064711 |
| nDCG@5 | +0.071057 |

结论：

- HSC-RAG 在 BM25 下相对 fixed 全指标提升。
- HSC-RAG + Hybrid 在主实验中取得最高 Recall@1：0.288690，说明其对首位证据定位有优势。
- recursive 的 Recall@5 较高，主要来自 overlap 扩大命中面，但其 `mixed_title_paths=36`，结构一致性弱于 HSC-RAG。
- semantic 粒度更细，但容易拆散证据链，nDCG@5 表现不稳定。

更完整的实验摘要见：

```text
reports\retrieval_eval_summary.md
```

## DuReader 补充实验

DuReader 补充实验验证了中文公开数据集上的完整链路：

```text
DuReader search.dev -> GovernedDocument
-> fixed / hsc_rag
-> BM25 / Dense FAISS / Hybrid
-> Recall@1/3/5、MRR、nDCG@5
```

运行分段：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_chunking.py --input data\processed\dureader\search_dev\governed_documents.jsonl --output-dir data\processed\dureader\search_dev --strategies fixed,hsc_rag --fixed-target 512 --fixed-overlap 64 --hsc-min 180 --hsc-target 512 --hsc-max 900
```

运行检索评估：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_retrieval_eval.py --chunk-dir data\processed\dureader\search_dev --gold-evidence data\processed\dureader\search_dev\gold_evidence.jsonl --strategies fixed,hsc_rag --retrievers bm25,dense,hybrid --top-k 1,3,5 --ndcg-k 5 --dense-encoder tfidf_svd --dense-svd-dim 128 --hybrid-alpha 0.55
```

DuReader 主要结果：

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | BM25 | 0.370748 | 0.738095 | 0.846939 | 0.662439 | 0.683983 |
| hsc_rag | BM25 | 0.241497 | 0.642857 | 0.870748 | 0.583528 | 0.642804 |
| fixed | Dense | 0.425170 | 0.642857 | 0.836735 | 0.671097 | 0.684329 |
| hsc_rag | Dense | 0.275510 | 0.632653 | 0.870748 | 0.613800 | 0.655386 |
| fixed | Hybrid | 0.465986 | 0.700680 | 0.826531 | 0.705782 | 0.711163 |
| hsc_rag | Hybrid | 0.272109 | 0.653061 | 0.853741 | 0.600761 | 0.646463 |

DuReader 的结论和 QASPER 不完全相同：fixed 在 Recall@1/MRR 上更强，因为 DuReader 的 gold evidence 是段落级标注，较细的 chunk 更容易首位命中；HSC-RAG 在结构质量和 Recall@5 上更好，例如 BM25 Recall@5 从 0.846939 提升到 0.870748，并将 `mixed_title_paths` 从 93 降到 42。完整说明见：

```text
reports\dureader_eval_summary.md
```

## 启动后端 API

```powershell
cd /d E:\practical_training\HSC_RAG\backend
& E:\anaconda3\envs\HSC_RAG\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```text
http://127.0.0.1:8000/api/health
```

主要 API：

```text
POST /api/v1/chunk
POST /api/v1/chunk/batch
POST /api/v1/agent/run
GET /api/overview
GET /api/metrics?retriever=bm25
GET /api/metrics?retriever=dense
GET /api/metrics?retriever=hybrid
GET /api/queries?retriever=bm25
GET /api/bad-cases?retriever=bm25
GET /api/queries/{query_id}/comparison?retriever=bm25
POST /api/cache/refresh
```

## 智能体标准接口

HSC-RAG 的在线交付接口是：

```text
POST http://127.0.0.1:8000/api/v1/chunk
```

接口定位：

- 调用方：上游转换流水线或数据治理流水线。
- 输入：结构化全文 `GovernedDocument`，即已经完成口径统一、结构化治理和来源锚定的文档对象。
- 输出：`RagChunk[]`，每个 chunk 含正文、长度、标题路径、来源块、原文回链、标签、摘要、实体标签和质量标记。
- 默认策略：`hsc_rag`。
- 可选策略：`fixed`、`recursive`、`semantic`、`hsc_rag`，用于实验对比。

当需要展示大模型参与时，使用 `POST /api/v1/agent/run` 并在 instruction 中明确要求“摘要、标签、实体标签、语义完整性评分或大模型语义组织”。LangChain agent 会调用 `chunk_and_enrich_current_document` / `chunk_and_enrich_current_batch` 工具：先运行确定性 HSC-RAG 分段，再调用 LLM semantic organization skill，将大模型结果写入 `metadata.llm_enrichment`。这样大模型承担的是任务书要求的内容组织与质量评价，而不是只做外层路由。

请求示例：

```json
{
  "strategy": "hsc_rag",
  "config": {
    "target_tokens": 512,
    "max_tokens": 900,
    "include_title_context": true
  },
  "include_report": true,
  "document": {
    "doc_id": "demo_doc_001",
    "dataset": "demo",
    "split": "test",
    "source_doc_id": "demo_doc_001",
    "title": "Demo Governed Document",
    "normalization_status": "provided_by_upstream",
    "blocks": [
      {
        "block_id": "demo_doc_001_p001",
        "doc_id": "demo_doc_001",
        "type": "paragraph",
        "text": "This is a governed paragraph ready for chunk packaging.",
        "order": 1,
        "level": 1,
        "title_path": ["Introduction"],
        "source_anchor": {
          "dataset": "demo",
          "split": "test",
          "source_doc_id": "demo_doc_001",
          "section_name": "Introduction"
        }
      }
    ]
  }
}
```

响应核心结构：

```json
{
  "agent": "hsc-rag",
  "strategy": "hsc_rag",
  "doc_id": "demo_doc_001",
  "chunk_count": 1,
  "chunks": [
    {
      "chunk_id": "demo_doc_001_hsc_rag_chunk_00001",
      "doc_id": "demo_doc_001",
      "strategy": "hsc_rag",
      "text": "[Introduction] This is a governed paragraph ready for chunk packaging.",
      "source_blocks": ["demo_doc_001_p001"],
      "source_anchor": {
        "source_doc_id": "demo_doc_001",
        "sections": ["Introduction"],
        "first_block_id": "demo_doc_001_p001",
        "last_block_id": "demo_doc_001_p001",
        "block_count": 1
      },
      "quality_flags": ["short_chunk", "source_anchor_complete", "title_path_consistent", "section_boundary_respected", "hsc_structure_aware"]
    }
  ],
  "report": {
    "input_contract": "GovernedDocument",
    "output_contract": "RagChunk[]",
    "governance_stage": "post_normalization_packaging"
  }
}
```

批量接口：

```text
POST http://127.0.0.1:8000/api/v1/chunk/batch
```

批量接口输入字段为 `documents: GovernedDocument[]`，输出每篇文档的 chunk 序列和总 chunk 数，适合转换流水线一次提交多篇结构化全文。

## 启动前端页面

第一次运行前安装依赖：

```powershell
cd /d E:\practical_training\HSC_RAG\frontend
npm.cmd install
```

启动开发服务器：

```powershell
npm.cmd run dev -- --port 5173
```

打开页面：

```text
http://127.0.0.1:5173
```

构建检查：

```powershell
npm.cmd run build
```

前端页面包含：

- 分段策略指标矩阵。
- BM25 / Dense / Hybrid 检索器切换。
- chunk 结构质量摘要。
- bad case query 列表。
- 同一 query 下 fixed / recursive / semantic / HSC-RAG 的 Top-5 检索结果对比。
- 命中 gold evidence 的 chunk 高亮显示。

更多启动说明见：

```text
docs\dashboard_usage.md
```

## 为什么值得研究

通用分段方法通常只关注长度、字符边界或局部语义相似度，容易出现以下问题：

- chunk 跨章节混合，破坏上下文口径一致性。
- 表格、图注、公式、列表等结构块被拆散。
- 检索命中后难以回溯到原文结构位置。
- 分段结果缺少面向下游消费的质量标记和来源锚点。

HSC-RAG 的攻关点在于：

- 在 `GovernedDocument` 层面消费治理后的结构化文档。
- 利用 `title_path`、block type、source anchor 进行结构感知分段。
- 对 table / figure / code / formula / list 等 protected block 尽量保持完整。
- 为每个 chunk 输出 `source_blocks`、`source_anchor`、`quality_flags`、`tags`、`summary` 等可解释字段。
- 用公开数据集的 question/evidence 标注自动评估分段对 RAG 检索的影响。

## 大模型相关技术使用说明

为满足课程“所有备选课题均需使用大模型相关技术”“围绕智能体 Agent 相关研究展开”的要求，本项目将 HSC-RAG 明确实现为“确定性结构感知分段 + 大模型语义组织”的混合式智能体。

核心分段边界不直接交给大模型决定，原因是课题任务书要求数据治理过程可追溯、可回放，并强调“能用确定性规则解决就不消耗大模型算力”。因此，HSC-RAG 使用确定性结构规则负责：

- 标题层级边界识别。
- 不破句和长度约束。
- table / figure / code / formula / list 等结构块保护。
- chunk 到原文 `source_blocks` 与 `source_anchor` 的完整回链。

大模型相关能力放在分段之后，作为 `LLM Semantic Organization Skill`：

- 对每个 chunk 生成忠实摘要。
- 生成主题标签与关键词。
- 抽取或补充实体标签。
- 评价 chunk 语义完整性。
- 评价摘要忠实度与标签准确性。
- 可选生成面向微调或评测的 QA/指令数据样例。

该模块默认支持离线可复现的 `mock` provider，便于无 API key、无外网或私有化部署场景下演示完整流程；后续可切换为 `openai_compatible` provider，接入 DeepSeek、通义千问、OpenAI、Ollama/vLLM 等兼容 OpenAI Chat Completions 协议的服务。

离线增强示例：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_llm_enrichment.py `
  --input data\processed\qasper\train\chunks_hsc_rag.jsonl `
  --output data\processed\qasper\train\chunks_hsc_rag_llm_enriched.jsonl `
  --provider mock `
  --limit 20 `
  --include-qa
```

输出文件：

```text
data\processed\qasper\train\chunks_hsc_rag_llm_enriched.jsonl
data\processed\qasper\train\hsc_rag_synthetic_qa.jsonl
reports\llm_enrichment_summary.md
reports\llm_enrichment_summary.json
```

真实大模型兼容接口示例：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_llm_enrichment.py `
  --input data\processed\qasper\train\chunks_hsc_rag.jsonl `
  --output data\processed\qasper\train\chunks_hsc_rag_llm_enriched.jsonl `
  --provider openai_compatible `
  --base-url https://api.deepseek.com/v1 `
  --model deepseek-chat `
  --api-key-env DEEPSEEK_API_KEY `
  --limit 20 `
  --include-qa
```

增强结果写入每个 chunk 的 `metadata.llm_enrichment` 字段，不改变原有 `RagChunk` 主结构，因此不会影响已完成的固定切分对比实验、BM25/Dense/Hybrid 检索评估和前端展示。

## 统一智能体流水线入口

为了便于演示和被上游流水线调用，项目提供统一入口：

```text
scripts\run_agent_pipeline.py
```

该脚本将原先分散的步骤串成一条 Agent Pipeline：

```text
结构化输入/Markdown
-> GovernedDocument
-> fixed / recursive / semantic / hsc_rag 分段
-> 可选 BM25 / Dense / Hybrid 检索评估
-> 可选 LLM 语义组织增强
-> agent_pipeline_summary.json / agent_pipeline_summary.md
```

对标准 `GovernedDocument` JSONL 运行完整分段与检索评估：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_agent_pipeline.py `
  --input data\processed\qasper\train\governed_documents.jsonl `
  --input-format governed_jsonl `
  --output-dir runs\qasper_agent_demo `
  --strategies fixed,recursive,semantic,hsc_rag `
  --run-eval `
  --gold-evidence data\processed\qasper\train\gold_evidence.jsonl `
  --retrievers bm25,dense,hybrid `
  --top-k 1,3,5 `
  --ndcg-k 5
```

如果只想演示“输入结构化全文，输出 chunk 序列”，可以不传 `--run-eval`：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_agent_pipeline.py `
  --input data\processed\qasper\train\governed_documents.jsonl `
  --input-format governed_jsonl `
  --output-dir runs\chunk_only_demo `
  --strategies hsc_rag
```

对 Markdown 运行分段演示：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_agent_pipeline.py `
  --input docs\demo.md `
  --input-format markdown `
  --output-dir runs\markdown_agent_demo `
  --strategies hsc_rag
```

注意：Markdown 输入可以生成分段、标签、摘要、实体标签和原文回链；但严格的 Recall@k / MRR / nDCG@5 需要额外提供 `gold_evidence.jsonl`，因为检索评估必须知道每个 query 的标准证据块。

带大模型语义组织增强的运行示例：

```powershell
$env:SILICONFLOW_API_KEY="你的硅基流动 API Key"

& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_agent_pipeline.py `
  --input data\processed\qasper\train\governed_documents.jsonl `
  --input-format governed_jsonl `
  --output-dir runs\qasper_agent_llm_demo `
  --strategies fixed,hsc_rag `
  --run-eval `
  --gold-evidence data\processed\qasper\train\gold_evidence.jsonl `
  --run-llm-enrichment `
  --llm-provider openai_compatible `
  --llm-base-url https://api.siliconflow.cn/v1 `
  --llm-model Qwen/Qwen3-VL-32B-Instruct `
  --llm-api-key-env SILICONFLOW_API_KEY `
  --llm-limit 20 `
  --llm-timeout-seconds 240 `
  --llm-max-input-chars 1200 `
  --llm-max-output-tokens 900 `
  --llm-disable-response-format
```

统一入口的主要输出：

```text
governed_documents.jsonl
chunks_hsc_rag.jsonl
chunk_report_hsc_rag.json
chunking_summary.json
retrieval_eval_multi_summary.json
chunks_hsc_rag_llm_enriched.jsonl
llm_enrichment_summary.json
agent_pipeline_summary.json
agent_pipeline_summary.md
```

## LangChain Agent API

The project exposes deterministic HSC-RAG chunking through a LangChain tool layer.
LangChain is used for agent orchestration and tool selection; chunk boundaries are
still produced by the local HSC-RAG implementation.

Offline CLI demo:

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_langchain_agent.py `
  --input data\processed\qasper\train\governed_documents.jsonl `
  --limit-docs 1 `
  --strategy hsc_rag `
  --provider mock `
  --output runs\langchain_agent_demo.json
```

Online endpoint:

```text
POST /api/v1/agent/run
```

Detailed usage:

```text
docs\langchain_agent_usage.md
```

SiliconFlow Qwen API demo fields for `POST /api/v1/agent/run`:

```json
{
  "preferred_tool": "chunk_and_enrich_current_document",
  "llm_provider": "openai_compatible",
  "llm_model": "Qwen/Qwen3-VL-32B-Instruct",
  "llm_base_url": "https://api.siliconflow.cn/v1",
  "llm_api_key_env": "SILICONFLOW_API_KEY",
  "llm_timeout_seconds": 240,
  "llm_use_response_format": false
}
```

`preferred_tool` makes the workflow explicit: local HSC-RAG creates auditable
chunks first, then the configured Qwen model performs semantic organization
inside `chunk_and_enrich_current_document`. A successful real-model call is
visible in `result.report.llm_semantic_organization.provider_execution_counts`
and in each chunk's `metadata.llm_enrichment.provider_execution =
remote_llm_call`.

## Topic 11 JSON Handoff Contract

Topic 11 is a callable specialist agent between Topic 4 and Topic 5:

```text
Topic 4 normalized structured JSON
-> Topic 11 HSC-RAG chunking and content organization
-> chunk sequence JSON
-> Topic 5 standard result package assembly
```

The formal online contract is JSON request to JSON response. JSONL files under
`data\processed\**` are offline batch experiment artifacts used for retrieval
evaluation and reproducibility.

Formal service endpoints:

```text
POST /api/v1/chunk
POST /api/v1/chunk/batch
POST /api/v1/agent/run
```

Contract and examples:

```text
docs\topic11_json_contract.md
docs\topic11_handoff_contract_zh.md
examples\topic11_request.json
examples\topic11_response.json
```

中文交付说明见 `docs\topic11_handoff_contract_zh.md`。该文档明确了课题 11 在数据治理链路中的边界：输入课题 4 或上游流程输出的 `GovernedDocument` JSON，输出课题 5 可调用和封装的 `RagChunk[]` JSON。

验证正式 JSON 契约：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_topic11_contract.py
```

运行正式 pytest 测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
& E:\anaconda3\envs\HSC_RAG\python.exe -m pytest backend\tests -p no:cacheprovider -q
```

当前测试覆盖：

- `test_chunk_contract.py`：验证 `examples/topic11_request.json` 的服务输出与 `examples/topic11_response.json` 保持一致。
- `test_hsc_rag_protected_blocks.py`：验证 table / code / formula 等 protected block 不被拆散。
- `test_source_anchor_complete.py`：验证 `source_blocks` 与 `source_anchor` 聚合字段完整一致。
