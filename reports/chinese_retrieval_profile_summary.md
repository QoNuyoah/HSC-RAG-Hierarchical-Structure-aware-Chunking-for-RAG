# 通用中文检索 Profile 说明

本文档记录通用中文检索预设 `zh_cjk`。该 profile 的目标不是重新切碎 HSC-RAG chunk，也不是针对某个业务词、某个样例 JSON 或某个公开数据集写规则，而是在检索阶段补齐中文连续文本的 tokenization 能力。

## Profile 边界

`zh_cjk` 在 `scripts/run_retrieval_eval.py` 中定义，当前只设置一个参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `tokenizer_profile` | `cjk_2_4gram` | 对连续 CJK 文本抽取 2-4 字符 ngram |

它不会改变以下内容：

- 不改变 HSC-RAG 分段算法。
- 不改变 `retrievers`，默认仍由命令行参数或脚本默认值决定。
- 不改变 `dense_svd_dim`、`hybrid_alpha`、`include_metadata` 等排序参数。
- 不包含业务词表、样例文档 ID、数据集字段判断或针对某个问题文本的分支逻辑。

因此，`zh_cjk` 是语言层面的检索配置，而不是面向 DuReader、QASPER 或某个验收样例的专用调参 profile。

## 通用机制

原 `mixed` tokenizer 对中文更接近单字 token。`cjk_2_4gram` 会把连续 CJK 片段扩展为 2/3/4 字符 ngram，使通用中文多字短语能够直接参与 BM25 和 TF-IDF/SVD 向量化。该机制只依赖 Unicode CJK 字符范围和固定 ngram 窗口，不依赖任何特定领域词表。

示例命令如下。这里的 DuReader 只是中文公开数据集评测样例，不是 profile 的适配对象：

```powershell
& E:\anaconda3\envs\HSC_RAG\python.exe scripts\run_retrieval_eval.py `
  --chunk-dir runs\chinese_chunking_sweep\hsc_cn_512_900 `
  --gold-evidence data\processed\dureader\search_dev\gold_evidence.jsonl `
  --output-dir runs\chinese_chunking_sweep\hsc_cn_512_900\eval_zh_cjk_profile `
  --strategies fixed,hsc_rag `
  --retrieval-profile zh_cjk `
  --top-k 1,3,5 `
  --ndcg-k 5
```

如果后续要比较不同 `dense_svd_dim`、`hybrid_alpha` 或检索器组合，应在实验命令中显式指定，并把它作为“实验参数搜索”记录在报告里，不应写入 `zh_cjk` 的默认 profile。这样可以保证老师随便输入一个符合契约的 JSON 文档时，主分段链路仍然是通用的。
