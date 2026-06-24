# DuReader 补充实验摘要

本实验用于验证 HSC-RAG 在中文公开数据集上的可运行性与跨数据集适配能力。DuReader 与 QASPER 不同：它是中文网页问答数据集，每条样本包含一个问题、多个候选网页文档和答案相关段落标注。因此本实验将每条 QA 样本封装为一篇 `GovernedDocument`，候选网页段落封装为 `GovernedBlock`，并将 `is_selected + most_related_para` 映射为 chunk 检索评估所需的 `gold_block_ids`。

## 数据与转换

| 项目 | 数值 |
|---|---:|
| 数据集 | DuReader v2.0 preprocessed |
| 子集 | `search.dev` |
| 抽样规模 | 50 条 QA 样本 |
| GovernedDocument | 50 |
| GovernedBlock | 1870 |
| Query | 50 |
| 可评估 Query | 49 |
| Gold evidence | 91 |
| Evidence 映射率 | 100.0% |

输出目录：

```text
data/processed/dureader/search_dev
```

核心产物：

```text
governed_documents.jsonl
blocks.jsonl
queries.csv
gold_evidence.jsonl
conversion_report.json
```

## 分段结果

| Strategy | Chunks | Avg Tokens | Max Tokens | Title Consistent | Mixed Title Paths |
|---|---:|---:|---:|---:|---:|
| fixed | 322 | 383.63 | 512 | 229 | 93 |
| hsc_rag | 293 | 587.17 | 900 | 251 | 42 |

结构观察：

- HSC-RAG 的 chunk 数量更少：293 vs 322，说明其通过结构封装减少了过细碎片。
- HSC-RAG 的跨标题混合显著更少：`mixed_title_paths` 从 93 降到 42，下降约 54.84%。
- HSC-RAG 的标题路径一致 chunk 更多：251 vs 229，说明其更能保持候选网页/段落结构边界。
- fixed 的最大长度严格控制在 512，适合作为定长切分基线；HSC-RAG 采用 `target_tokens=512, max_tokens=900`，用于保留同一候选网页内的相邻证据上下文。

## 检索结果

实验设置：

- 检索范围：same-doc，即每个问题只在该 QA 样本对应的候选网页 chunks 中检索。
- Gold：DuReader 的 `is_selected` 文档及其 `most_related_para`。
- 评价指标：Recall@1 / Recall@3 / Recall@5 / MRR / nDCG@5。
- Dense：本地 `TF-IDF + SVD + FAISS`，使用中英文混合 tokenizer，不依赖联网模型下载。

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | BM25 | 0.370748 | 0.738095 | 0.846939 | 0.662439 | 0.683983 |
| hsc_rag | BM25 | 0.241497 | 0.642857 | 0.870748 | 0.583528 | 0.642804 |
| fixed | Dense | 0.425170 | 0.642857 | 0.836735 | 0.671097 | 0.684329 |
| hsc_rag | Dense | 0.275510 | 0.632653 | 0.870748 | 0.613800 | 0.655386 |
| fixed | Hybrid | 0.465986 | 0.700680 | 0.826531 | 0.705782 | 0.711163 |
| hsc_rag | Hybrid | 0.272109 | 0.653061 | 0.853741 | 0.600761 | 0.646463 |

HSC-RAG 相对 fixed 的 Top-5 覆盖变化：

| Retriever | Recall@5 Delta | Full Recall@5 Delta |
|---|---:|---:|
| BM25 | +0.023809 | +0.040816 |
| Dense | +0.034013 | +0.081632 |
| Hybrid | +0.027210 | +0.061224 |

## 结论

DuReader 补充实验说明 HSC-RAG 可以在中文公开数据集上完整跑通 `GovernedDocument -> chunks -> retrieval evaluation` 链路，并且相对 fixed 定长切分具有更好的结构组织能力和 Top-5 证据覆盖能力。

需要同时说明的是，DuReader 的 gold evidence 是“答案相关段落”级标注，fixed 切分粒度更细，因此在 Recall@1、MRR、nDCG@5 上更容易把单个答案段落排到首位。HSC-RAG 的优势主要体现在结构边界、候选网页上下文保留、Top-5 证据覆盖和完整证据覆盖上。这与本课题“成果封装阶段的结构感知分段”定位是一致的：它不是单纯追求最小粒度命中，而是面向下游 RAG 消费输出结构一致、可追溯、上下文更完整的 chunks。
