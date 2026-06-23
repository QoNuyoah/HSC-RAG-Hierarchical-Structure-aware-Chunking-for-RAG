# QASPER -> GovernedDocument Adapter

本模块用于把本地 `QASPER.zip` 转换为 HSC-RAG 的标准输入 `GovernedDocument`。

## 课题边界

HSC-RAG 是课题 11“面向 RAG 的智能分段与内容组织智能体”，处于数据治理流水线的“口径已统一之后、成果封装之时”。因此 Adapter 不声称完成原始文档解析、通用清洗或术语归一，而是把公开数据集视为发布方已经整理好的标准化数据，并记录：

- `normalization_status = "provided_by_dataset"`
- `term_policy = "dataset_provided"`
- `governance_stage = "post_normalization_packaging"`

这样后续 HSC-RAG 分段不会偏到课题 2/3/4 的职责范围。

## 输出文件

运行转换脚本后会生成：

- `governed_documents.jsonl`：每行一个治理后结构化文档。
- `blocks.jsonl`：所有文档块，包含 heading、paragraph、abstract、table、figure 等。
- `queries.csv`：问题、答案、gold evidence block 映射。
- `gold_evidence.jsonl`：每个 query 的 evidence 匹配详情。
- `conversion_report.json`：转换统计和 warning。

## 快速运行

在项目根目录执行：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\convert_qasper.py --split train --limit-docs 5
```

输出目录：

```text
data\processed\qasper\train
```

## 后续接入

HSC-RAG chunker 后续应直接消费 `governed_documents.jsonl` 或 `blocks.jsonl`，并输出 RAG-ready chunks。

