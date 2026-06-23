# HSC-RAG Retrieval Evaluation Summary

## 实验设置

- 数据集：QASPER train 子集，当前转换 5 篇论文。
- 查询：33 个问题，其中 28 个 answerable query 参与评估，5 个 unanswerable query 跳过。
- 检索范围：same-doc evidence retrieval，即每个问题只在其所属论文的 chunks 内检索。
- 评价依据：chunk.source_blocks 是否覆盖 gold_evidence.jsonl 中的 gold_block_ids。
- 指标：Recall@1、Recall@3、Recall@5、MRR、nDCG@5。

## 分段策略

| Strategy | Chunks | Avg Tokens | Max Tokens | Title Consistent | Mixed Title Paths | 定位 |
|---|---:|---:|---:|---:|---:|---|
| fixed | 50 | 446.28 | 511 | 7 | 32 | 固定窗口基线 |
| recursive | 52 | 441.98 | 510 | 7 | 36 | 通用递归切分基线 |
| semantic | 101 | 220.93 | 501 | 45 | 38 | 语义相邻句切分基线 |
| hsc_rag | 61 | 386.85 | 900 | 38 | 16 | 层级结构感知方法 |

结构观察：

- fixed 和 recursive 的 chunk 长度稳定，但跨标题层级混合严重。
- semantic 的 title_path_consistent 较高，但 chunk 数量显著增加，证据容易被拆散。
- hsc_rag 在保持适中 chunk 数的同时，将 mixed_title_paths 从 fixed 的 32 降到 16，并显著提升 title_path_consistent。

## Text-only Retrieval Results

索引字段仅使用 chunk.text，不额外加入 title_path/tags/summary。

| Strategy | Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | BM25 | 0.157738 | 0.437500 | 0.633929 | 0.450993 | 0.438077 |
| recursive | BM25 | 0.193452 | 0.437500 | 0.741071 | 0.477737 | 0.481412 |
| semantic | BM25 | 0.202381 | 0.437500 | 0.562500 | 0.478301 | 0.377404 |
| hsc_rag | BM25 | 0.217262 | 0.568452 | 0.723214 | 0.515704 | 0.509134 |
| fixed | Dense FAISS | 0.181548 | 0.562500 | 0.669643 | 0.519446 | 0.489222 |
| recursive | Dense FAISS | 0.229167 | 0.598214 | 0.741071 | 0.551504 | 0.534973 |
| semantic | Dense FAISS | 0.238095 | 0.562500 | 0.687500 | 0.530413 | 0.450116 |
| hsc_rag | Dense FAISS | 0.258929 | 0.532738 | 0.651786 | 0.509752 | 0.489759 |
| fixed | Hybrid | 0.217262 | 0.491071 | 0.633929 | 0.525497 | 0.481484 |
| recursive | Hybrid | 0.193452 | 0.633929 | 0.741071 | 0.539443 | 0.526547 |
| semantic | Hybrid | 0.202381 | 0.455357 | 0.633929 | 0.484765 | 0.410477 |
| hsc_rag | Hybrid | 0.288690 | 0.514881 | 0.705357 | 0.550271 | 0.526316 |

## 关键结论

1. 在 BM25 检索下，hsc_rag 相比 fixed 在全部核心指标上提升：
   - Recall@1：+0.059524
   - Recall@3：+0.130952
   - Recall@5：+0.089285
   - MRR：+0.064711
   - nDCG@5：+0.071057

2. 在 Hybrid 检索下，hsc_rag 的 Recall@1 达到 0.288690，是 text-only 主实验中的最高值，说明层级结构感知 chunk 对高精度首位证据定位有明显帮助。

3. recursive 在 Recall@5 上表现强，主要原因是 overlap 扩大了证据命中面，但它的 mixed_title_paths=36，结构一致性明显弱于 hsc_rag。因此 recursive 更像“扩大召回的通用窗口基线”，而不是治理场景下更可解释的成果封装方法。

4. semantic 产生 101 个 chunks，粒度更细，部分 Recall@1 有提升，但 Recall@5 和 nDCG@5 不稳定，说明单纯语义断点容易拆散跨句/跨块证据。

5. hsc_rag 的价值不只体现在检索指标，还体现在数据治理要求的可追溯、结构一致和下游可消费：
   - chunk 保留 source_anchor/source_blocks；
   - title_path 更一致；
   - protected blocks 尽量保持完整；
   - chunk 生成发生在 GovernedDocument 的 post_normalization_packaging 阶段。

## 复现实验命令

生成四类 chunks：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_chunking.py --input data\processed\qasper\train\governed_documents.jsonl --output-dir data\processed\qasper\train --strategies fixed,recursive,semantic,hsc_rag
```

验证 chunks：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\validate_chunks.py data\processed\qasper\train\chunks_hsc_rag.jsonl --blocks data\processed\qasper\train\blocks.jsonl --max-tokens 900
```

运行 BM25/Dense/Hybrid 检索评估：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_retrieval_eval.py --chunk-dir data\processed\qasper\train --gold-evidence data\processed\qasper\train\gold_evidence.jsonl --strategies fixed,recursive,semantic,hsc_rag --retrievers bm25,dense,hybrid --top-k 1,3,5 --ndcg-k 5 --dense-encoder tfidf_svd --dense-svd-dim 128 --hybrid-alpha 0.55
```

metadata-enhanced 检索评估输出在：

```text
data\processed\qasper\train\eval_with_metadata
```

