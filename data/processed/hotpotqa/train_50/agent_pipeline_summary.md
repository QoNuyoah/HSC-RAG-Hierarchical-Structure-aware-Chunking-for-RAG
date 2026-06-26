# HSC-RAG Agent Pipeline Report

- Input: `data\processed\hotpotqa\train_50\governed_documents.jsonl`
- Output directory: `data\processed\hotpotqa\train_50`
- Documents: 50

## Steps

| Step | Status | Output |
|---|---|---|
| load_input | completed | `data\processed\hotpotqa\train_50\governed_documents.jsonl` |
| chunking | completed | `data\processed\hotpotqa\train_50\chunking_summary.json` |
| retrieval_evaluation | completed | `data\processed\hotpotqa\train_50\retrieval_eval_multi_summary.json` |

## Chunking
- `fixed`: chunks=115, avg_tokens=394.4, max_tokens=512, output=`data\processed\hotpotqa\train_50\chunks_fixed.jsonl`
- `hsc_rag`: chunks=217, avg_tokens=233.01, max_tokens=539, output=`data\processed\hotpotqa\train_50\chunks_hsc_rag.jsonl`

## Retrieval Evaluation

| Strategy | Retriever | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| fixed | bm25 | 0.746000 | 1.000000 | 1.000000 | 0.990000 | 0.991013 |
| fixed | dense | 0.686000 | 1.000000 | 1.000000 | 0.960000 | 0.964052 |
| fixed | hybrid | 0.726000 | 1.000000 | 1.000000 | 0.980000 | 0.983632 |
| hsc_rag | bm25 | 0.547667 | 0.904667 | 0.990000 | 0.940000 | 0.920642 |
| hsc_rag | dense | 0.502667 | 0.889667 | 0.995000 | 0.923333 | 0.899482 |
| hsc_rag | hybrid | 0.557667 | 0.889667 | 0.985000 | 0.956667 | 0.919199 |
