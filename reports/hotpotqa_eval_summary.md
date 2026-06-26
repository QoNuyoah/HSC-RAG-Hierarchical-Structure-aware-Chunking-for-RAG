# HotpotQA 进阶补充实验摘要

## 定位

HotpotQA 用作 HSC-RAG 的进阶补充实验，重点观察多跳问答场景下的证据组织、来源回链和失败边界。它不替代 QASPER 主实验，也不作为课题验收中“相对 fixed 提升 >= 10%”的主证据。

原因是 HotpotQA 的每条样本包含一个问题和多个候选 Wikipedia 文章，supporting facts 往往分散在不同文章标题下。固定窗口切分会把多个候选文章混在同一个大 chunk 中，在 same-doc 检索评估下容易一次命中多个 supporting facts；这会让 fixed 在 Recall@1、MRR 和 nDCG@5 上占到“混合大块”的便宜，但它的结构一致性和可解释性较弱。

## 数据转换

| 项目 | 数值 |
|---|---:|
| 数据集 | HotpotQA train v1.1 |
| 抽样规模 | 50 条 QA 样本 |
| GovernedDocument | 50 |
| GovernedBlock | 2066 |
| Query | 50 |
| Gold evidence | 130 |
| Evidence 映射率 | 100.0% |
| Context articles | 500 |

转换产物：

```text
data/processed/hotpotqa/train_50
```

核心文件：

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
| fixed | 115 | 394.40 | 512 | 7 | 108 |
| hsc_rag | 217 | 233.01 | 539 | 40 | 177 |

结构观察：

- fixed 的 `mixed_title_paths=108/115`，说明绝大多数 chunk 混合了多个候选文章标题路径。
- HSC-RAG 的 chunk 数更多，原因是它优先尊重候选文章标题边界和句子/块级来源锚点。
- HotpotQA 的问题上下文由多个短文章组成，天然会造成多标题候选证据并列；因此它更适合做多跳问答边界分析，而不是作为 HSC-RAG 检索提升的主证明。

## Text-only 检索结果

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | BM25 | 0.746000 | 1.000000 | 1.000000 | 0.990000 | 0.991013 |
| fixed | Dense | 0.686000 | 1.000000 | 1.000000 | 0.960000 | 0.964052 |
| fixed | Hybrid | 0.726000 | 1.000000 | 1.000000 | 0.980000 | 0.983632 |
| hsc_rag | BM25 | 0.547667 | 0.904667 | 0.990000 | 0.940000 | 0.920642 |
| hsc_rag | Dense | 0.502667 | 0.889667 | 0.995000 | 0.923333 | 0.899482 |
| hsc_rag | Hybrid | 0.557667 | 0.889667 | 0.985000 | 0.956667 | 0.919199 |

## Metadata-enhanced 检索结果

索引字段包括：

```text
chunk.text + title_path + tags + summary
```

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | BM25 | 0.726000 | 1.000000 | 1.000000 | 0.980000 | 0.983632 |
| fixed | Dense | 0.716000 | 1.000000 | 1.000000 | 0.980000 | 0.982026 |
| fixed | Hybrid | 0.696000 | 1.000000 | 1.000000 | 0.970000 | 0.976250 |
| hsc_rag | BM25 | 0.561000 | 0.891333 | 0.990000 | 0.951667 | 0.924317 |
| hsc_rag | Dense | 0.519333 | 0.887667 | 0.995000 | 0.916667 | 0.904415 |
| hsc_rag | Hybrid | 0.581000 | 0.867667 | 1.000000 | 0.976667 | 0.935508 |

## 结论

HotpotQA 进阶实验说明：

- HSC-RAG 可以完整支持多跳问答数据集的 `GovernedDocument -> chunk -> retrieval evaluation` 流程。
- HotpotQA 的 `supporting_facts` 能稳定映射为 `gold_block_ids`，证据映射率为 100%。
- HSC-RAG 在 metadata-enhanced Hybrid 检索下达到 `Recall@5=1.000000`，说明 Top-5 范围内可以完整覆盖多跳证据。
- fixed 在 HotpotQA 上的 Recall@1/MRR/nDCG@5 更高，主要来自固定窗口跨候选文章混合，容易把多个 supporting facts 打包进同一个 chunk。
- 因此 HotpotQA 应作为“进阶补充与边界分析”：它证明系统可扩展到多跳 QA 数据，但不应替代 QASPER 主实验中的检索提升证据。

答辩表述建议：

> HotpotQA 补充实验用于验证智能体对多跳问答公开数据集的适配能力。实验显示，HSC-RAG 能将 supporting facts 稳定映射为可回溯证据块，并在 metadata-enhanced Hybrid 检索下达到 Recall@5=1.0。但由于 HotpotQA 的 same-doc 评估容易奖励 fixed chunk 的跨文章混合，fixed 在部分排序指标上更高。因此我们将 HotpotQA 作为进阶边界分析，主验收提升证据仍以 QASPER 为准。

## 复现命令

转换 HotpotQA：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\convert_hotpotqa.py `
  --zip HotpotQA.zip `
  --split train `
  --limit-docs 50 `
  --output-dir data\processed\hotpotqa\train_50
```

运行统一 Agent Pipeline：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_agent_pipeline.py `
  --input data\processed\hotpotqa\train_50\governed_documents.jsonl `
  --input-format governed_jsonl `
  --output-dir data\processed\hotpotqa\train_50 `
  --strategies fixed,hsc_rag `
  --run-eval `
  --gold-evidence data\processed\hotpotqa\train_50\gold_evidence.jsonl `
  --retrievers bm25,dense,hybrid `
  --top-k 1,3,5 `
  --ndcg-k 5 `
  --dense-encoder tfidf_svd `
  --dense-svd-dim 128 `
  --hybrid-alpha 0.55
```
运行 metadata-enhanced 评估：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_retrieval_eval.py `
  --chunk-dir data\processed\hotpotqa\train_50 `
  --gold-evidence data\processed\hotpotqa\train_50\gold_evidence.jsonl `
  --output-dir data\processed\hotpotqa\train_50\eval_with_metadata `
  --strategies fixed,hsc_rag `
  --retrievers bm25,dense,hybrid `
  --top-k 1,3,5 `
  --ndcg-k 5 `
  --include-metadata `
  --dense-encoder tfidf_svd `
  --dense-svd-dim 128 `
  --hybrid-alpha 0.55
```
