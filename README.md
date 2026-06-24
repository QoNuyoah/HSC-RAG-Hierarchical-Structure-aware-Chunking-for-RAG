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
    ├── run_chunking.py
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

