from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

from app.retrievers.bm25 import BM25ChunkRetriever, tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_retrieval_eval import apply_retrieval_profile  # noqa: E402

CJK_SAMPLE = "".join(chr(codepoint) for codepoint in range(0x4E00, 0x4E06))
CJK_OTHER_SAMPLE = "".join(chr(codepoint) for codepoint in range(0x4E10, 0x4E16))


def test_cjk_bigram_profile_emits_overlapping_cjk_ngrams():
    tokens = tokenize(CJK_SAMPLE, profile="cjk_bigram")

    assert CJK_SAMPLE[:2] in tokens
    assert CJK_SAMPLE[1:3] in tokens
    assert CJK_SAMPLE[:3] in tokens
    assert CJK_SAMPLE[:4] not in tokens


def test_cjk_2_4gram_profile_emits_longer_overlapping_cjk_ngrams():
    tokens = tokenize(CJK_SAMPLE, profile="cjk_2_4gram")

    assert CJK_SAMPLE[:2] in tokens
    assert CJK_SAMPLE[:3] in tokens
    assert CJK_SAMPLE[:4] in tokens
    assert CJK_SAMPLE[1:5] in tokens
    assert CJK_SAMPLE[:1] not in tokens


def test_jieba_profile_keeps_words_and_cjk_ngrams():
    tokens = tokenize("中文文本测试中文分词", profile="jieba")

    assert "中文" in tokens
    assert "文本" in tokens
    assert "测试" in tokens


def test_bm25_retriever_uses_chinese_tokenizer_profile():
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": CJK_SAMPLE,
            "source_blocks": ["b1"],
            "title_path": [CJK_SAMPLE[:4]],
            "token_count": 32,
            "quality_flags": [],
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "text": CJK_OTHER_SAMPLE,
            "source_blocks": ["b2"],
            "title_path": [CJK_OTHER_SAMPLE[:4]],
            "token_count": 18,
            "quality_flags": [],
        },
    ]
    retriever = BM25ChunkRetriever(chunks, tokenizer_profile="cjk_2_4gram")

    hits = retriever.search(CJK_SAMPLE[:4], top_k=2, doc_id="d1")

    assert hits[0].chunk_id == "c1"


def test_zh_cjk_retrieval_profile_only_switches_tokenization():
    args = Namespace(
        retrieval_profile="zh_cjk",
        retrievers="bm25,dense,hybrid",
        tokenizer_profile="mixed",
        dense_encoder="tfidf_svd",
        dense_svd_dim=128,
        hybrid_alpha=0.55,
        include_metadata=False,
    )

    profiled = apply_retrieval_profile(args)

    assert profiled.retrievers == "bm25,dense,hybrid"
    assert profiled.tokenizer_profile == "cjk_2_4gram"
    assert profiled.dense_encoder == "tfidf_svd"
    assert profiled.dense_svd_dim == 128
    assert profiled.hybrid_alpha == 0.55
    assert profiled.include_metadata is False
