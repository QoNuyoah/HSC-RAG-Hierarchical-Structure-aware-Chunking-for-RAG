# HSC-RAG 大模型语义组织增强报告

## 目的

本报告用于补充课题 11“面向 RAG 的智能分段与内容组织智能体”中的大模型相关能力。HSC-RAG 的分段边界仍由确定性结构感知算法完成，原因是边界控制、长度约束、原文回链和表格/公式保护更需要稳定、可追溯、可回放；大模型用于分段后的语义组织 Skill，包括 chunk 摘要、主题标签、实体标签、语义完整性评价、摘要忠实度评价和可选 QA/指令数据合成。

## 运行配置

| 项目 | 值 |
|---|---|
| 输入文件 | `data\processed\qasper\train\chunks_hsc_rag.jsonl` |
| 输出文件 | `runs\llm_enrich_demo\chunks_enriched.jsonl` |
| QA 输出 | `runs\llm_enrich_demo\hsc_rag_synthetic_qa.jsonl` |
| Provider | `mock` |
| Model | `mock-semantic-organizer-v1` |
| Prompt Version | `hsc-rag-enrich-v1` |
| 处理 chunk 数 | 5 |
| QA/指令样例数 | 5 |

## 指标摘要

| 指标 | 均值 |
|---|---:|
| 语义完整性评分 /5 | 4.70 |
| 摘要忠实度评分 /5 | 4.70 |
| 标签准确性评分 /5 | 4.50 |

faithfulness risk 分布：

```json
{
  "low": 4,
  "medium": 1
}
```

Provider execution 分布：

```json
{
  "mock_offline_replay": 5
}
```

## 样例

| chunk_id | 风险 | 语义完整 | 摘要忠实 | 标签 | 摘要 |
|---|---|---:|---:|---|---|
| qasper_train_1909.00694_hsc_rag_chunk_00001 | medium | 4.3 | 4.7 | Abstract, Introduction, paragraph, polarity | [Abstract] Recognizing affective events that trigger positive or negative sentiment has a wide range of ... |
| qasper_train_1909.00694_hsc_rag_chunk_00002 | low | 4.9 | 4.7 | Related Work, paragraph, sentiment, events | [Related Work] Learning affective events is closely related to sentiment analysis. |
| qasper_train_1909.00694_hsc_rag_chunk_00003 | low | 4.7 | 4.7 | Proposed Method, paragraph, event, polarity | [Proposed Method > Polarity Function] Our goal is to learn the polarity function $p(x)$, which predicts the ... |
| qasper_train_1909.00694_hsc_rag_chunk_00004 | low | 4.7 | 4.7 | Proposed Method, paragraph, event, loss | [Proposed Method > Discourse Relation-Based Event Pairs > AL (Automatically Labeled Pairs)] The seed lexicon ... |
| qasper_train_1909.00694_hsc_rag_chunk_00005 | low | 4.9 | 4.7 | Experiments, Dataset, AL, CA, and CO, paragraph | [Experiments > Dataset > AL, CA, and CO] As a raw corpus, we used a Japanese web corpus that was compiled ... |

## 与任务书要求的对应关系

- 内容打标：输出 `metadata.llm_enrichment.topic_tags`，作为大模型辅助主题/关键词标签。
- 摘要生成：输出 `metadata.llm_enrichment.summary`，并给出 `summary_faithfulness_score`。
- 实体标签：输出 `metadata.llm_enrichment.entity_tags`，用于补充规则实体标签。
- 语义完整：输出 `semantic_integrity_score` 和 `quality_reason`，用于人工抽样评价和 bad case 分析。
- 可私有化部署：`mock` 模式可离线复现；`openai_compatible` 模式后续可接入外部或本地兼容大模型服务。
- 不替代核心分段：大模型只负责语义组织，不负责不可控地改写分段边界，保证 chunk 的来源锚点和治理链路可追溯。
