# HSC-RAG 验收指标对照表

本报告由 `scripts/validate_acceptance_metrics.py` 自动生成，用于对照课题任务书中的可度量验收项。

- Strategy: `hsc_rag`
- Chunks: `62`
- Content blocks: `320`
- Protected blocks: `32`

| 指标项 | 验收口径 | 当前值 | 状态 |
|---|---|---:|---|
| 期望输出字段完整 | 每个 chunk 含文本、长度、标签、摘要、实体标签、原文锚点回链等字段 | 100.0% | 达标 |
| 不破句率 | 以 GovernedBlock 为最小治理内容单元，检查内容块未被跨 chunk 人工截断 | 100.0% | 达标 |
| 表格/公式/代码整体成块率 | protected block 在 HSC-RAG chunk 中完整出现一次 | 100.0% | 达标 |
| 目标长度区间命中率 | quality_flags 中 length_ok 的 chunk 占比 | 96.77% | 达标 |
| 原文回链完整率 | source_anchor 与 source_blocks 完整一致 | 100.0% | 达标 |
| 下游检索提升 | HSC-RAG 相对 fixed 的 BM25 Recall@5 与 nDCG@5 相对提升均 >= 10% | Recall@5 +11.27%; nDCG@5 +10.72% | 达标 |
| 结构/语义边界可解释 | 每个由 HSC-RAG 主算法收束的 chunk 记录 boundary_score、结构信号、语义距离、长度压力和 split_reason | 覆盖 91.9%，平均边界分 0.7312，平均语义距离 0.5673，语义触发 3 次 | 达标 |
| 语义完整 | 人工抽样评价 chunk 是否围绕同一主题、上下文是否足够、是否无明显断裂 | 抽样 20 个 chunks，均分 4.67/5，最低 4.0/5 | 达标 |
| 打标与摘要 | 人工抽样评价标签准确率、实体标签可用性与摘要忠实度 | 抽样 20 个 chunks，均分 4.22/5，最低 3.5/5 | 达标 |

## 说明

- “不破句率”采用工程可验证口径：HSC-RAG 以 `GovernedBlock` 为最小治理内容单元进行封装；当前样本中每个内容块仅出现在一个 HSC-RAG chunk 中，说明没有发生跨 chunk 的句中/块内截断。
- “期望输出字段完整”检查每个 chunk 是否包含 `text/token_count/tags/summary/entity_tags/source_blocks/source_anchor/quality_flags` 等下游消费必需字段。
- “结构/语义边界可解释”读取 HSC-RAG chunk metadata 中的 `closing_boundary_decision`，用于证明主算法不是纯固定规则，而是综合标题结构、语义距离和长度压力进行边界决策。
- “语义完整”和“打标与摘要”读取 `reports/manual_eval_hsc_rag.csv`。建议抽样 20 个 chunks；均分 >= 4.0 且最低分 >= 3.0 记为达标。
