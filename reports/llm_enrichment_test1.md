# HSC-RAG 大模型语义组织增强报告

## 目的

本报告用于补充课题 11“面向 RAG 的智能分段与内容组织智能体”中的大模型相关能力。HSC-RAG 的分段边界仍由确定性结构感知算法完成，原因是边界控制、长度约束、原文回链和表格/公式保护更需要稳定、可追溯、可回放；大模型用于分段后的语义组织 Skill，包括 chunk 摘要、主题标签、实体标签、语义完整性评价、摘要忠实度评价和可选 QA/指令数据合成。

## 运行配置

| 项目 | 值 |
|---|---|
| 输入文件 | `data\processed\qasper\train\chunks_hsc_rag.jsonl` |
| 输出文件 | `data\processed\qasper\train\chunks_hsc_rag_llm_enriched_test1.jsonl` |
| QA 输出 | `未启用` |
| Provider | `openai_compatible` |
| Model | `Qwen/Qwen3-VL-32B-Instruct` |
| Prompt Version | `hsc-rag-enrich-v1` |
| 处理 chunk 数 | 1 |
| QA/指令样例数 | 0 |

## 指标摘要

| 指标 | 均值 |
|---|---:|
| 语义完整性评分 /5 | 4.80 |
| 摘要忠实度评分 /5 | 5.00 |
| 标签准确性评分 /5 | 4.90 |

faithfulness risk 分布：

```json
{
  "low": 1
}
```

Provider execution 分布：

```json
{
  "remote_llm_call": 1
}
```

## 样例

| chunk_id | 风险 | 语义完整 | 摘要忠实 | 标签 | 摘要 |
|---|---|---:|---:|---|---|
| qasper_train_1909.00694_hsc_rag_chunk_00001 | low | 4.8 | 5.0 | affective event recognition, sentiment polarity, discourse relations, natural language processing | The paper addresses the challenge of recognizing affective events with positive or negative sentiment in ... |

## 与任务书要求的对应关系

- 内容打标：输出 `metadata.llm_enrichment.topic_tags`，作为大模型辅助主题/关键词标签。
- 摘要生成：输出 `metadata.llm_enrichment.summary`，并给出 `summary_faithfulness_score`。
- 实体标签：输出 `metadata.llm_enrichment.entity_tags`，用于补充规则实体标签。
- 语义完整：输出 `semantic_integrity_score` 和 `quality_reason`，用于人工抽样评价和 bad case 分析。
- 可私有化部署：`mock` 模式可离线复现；`openai_compatible` 模式后续可接入外部或本地兼容大模型服务。
- 不替代核心分段：大模型只负责语义组织，不负责不可控地改写分段边界，保证 chunk 的来源锚点和治理链路可追溯。
