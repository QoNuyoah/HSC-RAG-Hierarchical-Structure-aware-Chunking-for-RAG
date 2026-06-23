# HSC-RAG

HSC-RAG 是“面向 RAG 的智能分段与内容组织智能体”。本项目严格按照课题 11 的数据治理定位实现：分段发生在上游解析、清洗、归一之后，处于成果封装阶段，目标是把治理后、口径已统一的结构化全文组织为可直接供 RAG 消费的 chunk。

## 当前已完成

- 定义 `GovernedDocument`、`GovernedBlock`、`GovernedQuery` 等核心输入规范。
- 实现 `QASPER -> GovernedDocument` Adapter。
- 输出 `blocks.jsonl`、`queries.csv`、`gold_evidence.jsonl` 等后续 HSC-RAG 分段和检索评测所需文件。
- 提供转换结果验证脚本。

## 课题边界

HSC-RAG 不负责：

- 原始 PDF/Word/OCR 通用解析。
- 通用数据清洗。
- 术语归一和实体链接。
- 大模型或 embedding 模型训练。

HSC-RAG 负责：

- 接收治理后、口径已统一的结构化全文。
- 进行结构/语义感知分段。
- 生成 title_path、source_blocks、source_anchor、tags、summary、entity_tags、quality_flags。
- 用公开数据集 question/evidence 评估 Recall@k、MRR、nDCG。

## QASPER 转换

在项目根目录执行：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\convert_qasper.py --split train --limit-docs 5
```

输出目录：

```text
data\processed\qasper\train
```

输出文件：

- `governed_documents.jsonl`
- `blocks.jsonl`
- `queries.csv`
- `gold_evidence.jsonl`
- `conversion_report.json`

## 验证转换结果

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_governed_outputs.py data\processed\qasper\train
```

验证通过时，`status` 应为 `passed`。

## 下一步

1. 实现 fixed-size chunker，先跑通 `GovernedDocument -> chunks`。
2. 实现 HSC-RAG chunker，输出带 `title_path/source_anchor/quality_flags` 的 RAG-ready chunks。
3. 实现 BM25 / FAISS / Hybrid 检索。
4. 基于 QASPER 的 `gold_evidence.jsonl` 计算 Recall@k、MRR、nDCG。

