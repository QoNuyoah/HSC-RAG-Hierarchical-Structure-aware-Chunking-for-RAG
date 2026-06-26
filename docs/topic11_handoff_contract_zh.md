# 课题 11 JSON 交付契约与完成说明

## 1. 课题定位

本项目对应课题 11：面向 RAG 的智能分段与内容组织智能体。

在整条数据治理流水线中，课题 11 是一个可被调用的中间环节，不是原始文档解析器，也不是术语清洗或最终格式封装模块。它的职责是把上游已经治理好的结构化全文组织成适合 RAG 检索、索引和生成消费的 chunk 序列。

推荐的协作链路如下：

```text
课题 4：术语规范化 / 口径统一
-> 结构化、可追溯、已规范化的 GovernedDocument JSON
-> 课题 11：HSC-RAG 智能分段与内容组织
-> RagChunk[] JSON chunk 序列
-> 课题 5：格式标准化、目标格式转换、最终结果包封装
```

因此，本项目对外的正式输入输出都是 JSON。仓库中 `data/processed/**` 下的 JSONL 文件用于离线批处理、实验复现和检索评估，不是课题 5 调用课题 11 时必须采用的在线接口格式。

## 2. 职责边界

课题 11 负责：

- 接收课题 4 或上游数据治理流程产出的 `GovernedDocument` JSON。
- 按标题层级、块类型、来源锚点、实体标签等结构信息进行层级结构感知分段。
- 输出可直接用于 RAG 索引的 `RagChunk[]`。
- 保留 `source_blocks`、`source_anchor`、`title_path`、`entity_tags` 等可追溯信息。
- 生成 `summary`、`tags`、`quality_flags` 和运行报告，便于课题 5 封装和审计。
- 通过 LangChain 工具层暴露智能体式调用方式，但核心分段结果仍由本地确定性 HSC-RAG 方法产生。

课题 11 不负责：

- PDF、Word、网页等原始文件解析。
- OCR、版面还原、表格抽取等原始数据工程。
- 术语规范化、别名合并、单位换算、日期归一化。
- 面向最终提交物的 Markdown、HTML、压缩包、校验清单、manifest 等封装。
- 把课题 5 的目标格式字段做最终映射。

这些工作分别属于上游治理环节、课题 4 或课题 5 的职责。

## 3. 输入 JSON

单文档在线接口的请求模型是 `ChunkAgentRequest`：

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
    "min_tokens": 20,
    "target_tokens": 120,
    "max_tokens": 260
  },
  "include_report": true
}
```

完整可运行示例见：

```text
examples/topic11_request.json
```

### document 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | string | 数据治理流水线内部稳定文档 id。 |
| `dataset` | string | 来源数据集、项目批次或上游任务名。 |
| `split` | string | 数据集切分、批次或交接阶段。 |
| `source_doc_id` | string | 原始上游文档 id。 |
| `title` | string | 上游治理后的文档标题。 |
| `normalization_status` | enum | `provided_by_upstream` 表示由课题 4 或上游系统完成规范化。 |
| `term_policy` | string | 术语规范策略名称，用于说明口径来源。 |
| `governance_stage` | string | 建议为 `post_normalization_packaging`，表示已过规范化、等待内容组织。 |
| `blocks` | array | 结构化内容块，是 HSC-RAG 真正分段的输入。 |

### block 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `block_id` | string | 上游稳定内容块 id。 |
| `doc_id` | string | 所属文档 id。 |
| `type` | enum | 内容块类型，如 `paragraph`、`table`、`figure`、`code`、`formula`。 |
| `text` | string | 已规范化文本。课题 11 不再改写术语口径。 |
| `order` | integer | 上游解析和治理后的原文顺序。 |
| `level` | integer | 标题层级或结构深度。 |
| `title_path` | string[] | 标题路径，供结构感知分段使用。 |
| `source_anchor` | object | 回指原始来源的锚点，供课题 5 或前端追溯。 |
| `entity_tags` | string[] | 上游识别或规范化后的实体标签。 |
| `metadata` | object | 上游保留的扩展元数据。 |

## 4. 输出 JSON

单文档在线接口的响应模型是 `ChunkAgentResponse`：

```json
{
  "agent": "hsc-rag",
  "strategy": "hsc_rag",
  "doc_id": "topic11_demo_doc",
  "chunks": [],
  "chunk_count": 0,
  "report": {}
}
```

完整可运行示例见：

```text
examples/topic11_response.json
```

### chunk 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | string | 稳定 chunk id。 |
| `doc_id` | string | 所属文档 id。 |
| `dataset` | string | 来源数据集、项目批次或上游任务名。 |
| `split` | string | 数据集切分、批次或交接阶段。 |
| `strategy` | enum | 分段策略，正式交付推荐 `hsc_rag`。 |
| `text` | string | 下游 RAG 索引和检索使用的 chunk 文本。 |
| `token_count` | integer | 估算 token 长度。 |
| `title_path` | string[] | chunk 对应的结构标题路径。 |
| `source_blocks` | string[] | chunk 覆盖的上游 block id 列表。 |
| `source_anchor` | object | 聚合后的来源范围、章节、资产文件等追溯信息。 |
| `tags` | string[] | 面向检索组织的主题、结构和实体标签。 |
| `summary` | string | 对 chunk 内容的忠实摘要。 |
| `entity_tags` | string[] | 保留和补充后的实体标签。 |
| `quality_flags` | string[] | 质量标记，如 `source_anchor_complete`、`title_path_consistent`。 |
| `metadata` | object | 分段参数和扩展元数据。 |

## 5. 在线接口

正式推荐给课题 5 调用的接口：

```text
POST /api/v1/chunk
Content-Type: application/json
```

批量调用接口：

```text
POST /api/v1/chunk/batch
Content-Type: application/json
```

LangChain 智能体包装接口：

```text
POST /api/v1/agent/run
Content-Type: application/json
```

说明：

- `/api/v1/chunk` 和 `/api/v1/chunk/batch` 是最稳定的工程集成接口。
- `/api/v1/agent/run` 展示“老师建议用 LangChain 完成”的智能体编排形式。
- LangChain 层负责工具选择和智能体式交互，分段边界和 chunk 内容仍由确定性 HSC-RAG 工具产生。
- API key 不写入代码、请求示例或仓库文件。远程模型模式只接受环境变量名，如 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`。

## 6. 当前完成情况

核心交付已经完成：

- 已定义 `GovernedDocument`、`GovernedBlock`、`RagChunk`、`ChunkAgentRequest`、`ChunkAgentResponse` 等 Pydantic JSON 契约。
- 已实现 `POST /api/v1/chunk` 单文档 JSON 输入输出接口。
- 已实现 `POST /api/v1/chunk/batch` 批量 JSON 输入输出接口。
- 已实现 `POST /api/v1/agent/run` LangChain 包装接口。
- 已提供 `examples/topic11_request.json` 和 `examples/topic11_response.json`。
- 已提供 `docs/topic11_json_contract.md` 英文契约说明和本中文说明。
- 已通过公开数据集 QASPER、DuReader、HotpotQA 做离线分段和检索评估。
- 已有 FastAPI 后端和 React 前端实验看板，可展示指标与 bad case。

仍建议在最终提交前补充或确认：

- 与课题 5 的真实调用方做一次联调，确认字段名是否需要课题 5 侧适配。
- 在最终报告中放入接口调用截图、示例 JSON、前端看板截图和评估结果表。
- 如果老师要求必须展示大模型参与，可使用 `llm_provider=openai_compatible` 通过环境变量调用远程模型；如果不要求联网，默认 `mock` 模式已经能展示 LangChain 工具调用链。

结论：从课题 11 的代码功能、JSON 输入输出契约、LangChain 包装和可复现评估看，核心任务已经完成。剩余主要是和课题 5 的实际联调、报告整理、演示材料截图，而不是核心功能缺失。

## 7. 验证命令

在项目根目录运行：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_topic11_contract.py
```

期望看到：

```text
OK request schema: examples/topic11_request.json
OK service output matches: examples/topic11_response.json
OK FastAPI endpoint: /api/v1/chunk
OK LangChain endpoint: /api/v1/agent/run
```

也可以启动服务后直接调用：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe -m uvicorn app.main:app `
  --app-dir backend `
  --host 127.0.0.1 `
  --port 8000
```

```powershell
$body = Get-Content examples\topic11_request.json -Raw -Encoding UTF8
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/chunk `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```
