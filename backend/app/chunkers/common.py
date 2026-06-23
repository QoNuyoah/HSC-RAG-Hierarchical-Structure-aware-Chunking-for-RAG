# -*- coding: utf-8 -*-
"""Shared utilities for chunking GovernedDocument objects."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.core.schemas import ChunkSourceAnchor, GovernedBlock, GovernedDocument, RagChunk


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
CONTENT_BLOCK_TYPES = {"abstract", "paragraph", "list", "table", "figure", "code", "formula", "caption"}
PROTECTED_BLOCK_TYPES = {"table", "figure", "code", "formula", "list"}
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "into",
    "using",
    "based",
    "paper",
    "model",
    "method",
    "results",
    "approach",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def estimate_tokens(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    # QASPER is English-heavy but may contain Japanese/Chinese examples.
    # Use a mixed estimate instead of switching the whole paragraph to char count.
    cjk_chars = CJK_RE.findall(text)
    non_cjk = CJK_RE.sub(" ", text)
    latin_tokens = re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?", non_cjk)
    symbol_tokens = re.findall(r"[$€£¥]|[=<>≤≥±×÷]", non_cjk)
    return max(1, len(latin_tokens) + len(cjk_chars) + len(symbol_tokens))


def word_chunks(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) <= max_tokens:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    step = max(1, max_tokens - max(0, overlap_tokens))
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def sentence_chunks(text: str, max_tokens: int) -> list[str]:
    text = normalize_text(text)
    if estimate_tokens(text) <= max_tokens:
        return [text] if text else []
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return word_chunks(text, max_tokens=max_tokens, overlap_tokens=0)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if sentence_tokens > max_tokens:
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = []
                buffer_tokens = 0
            chunks.extend(word_chunks(sentence, max_tokens=max_tokens, overlap_tokens=0))
            continue
        if buffer and buffer_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(buffer))
            buffer = [sentence]
            buffer_tokens = sentence_tokens
        else:
            buffer.append(sentence)
            buffer_tokens += sentence_tokens
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def content_blocks(doc: GovernedDocument) -> list[GovernedBlock]:
    return [block for block in sorted(doc.blocks, key=lambda b: b.order) if block.type in CONTENT_BLOCK_TYPES and normalize_text(block.text)]


def block_text(block: GovernedBlock, include_title_path: bool = False) -> str:
    text = normalize_text(block.text)
    if not include_title_path or not block.title_path:
        return text
    title = " > ".join(block.title_path)
    if title and title.lower() not in text[:120].lower():
        return f"[{title}] {text}"
    return text


def common_title_path(blocks: list[GovernedBlock]) -> list[str]:
    paths = [block.title_path for block in blocks if block.title_path]
    if not paths:
        return []
    prefix: list[str] = []
    for items in zip(*paths):
        if all(item == items[0] for item in items):
            prefix.append(items[0])
        else:
            break
    if prefix:
        return prefix
    top_levels: list[str] = []
    for path in paths:
        if path and path[0] not in top_levels:
            top_levels.append(path[0])
    return top_levels[:4]


def aggregate_source_anchor(doc: GovernedDocument, blocks: list[GovernedBlock]) -> ChunkSourceAnchor:
    sections: list[str] = []
    assets: list[str] = []
    for block in blocks:
        section = block.source_anchor.section_name
        if section and section not in sections:
            sections.append(section)
        asset = block.source_anchor.asset_file
        if asset and asset not in assets:
            assets.append(asset)
    return ChunkSourceAnchor(
        dataset=doc.dataset,
        split=doc.split,
        source_doc_id=doc.source_doc_id,
        sections=sections,
        first_block_id=blocks[0].block_id if blocks else None,
        last_block_id=blocks[-1].block_id if blocks else None,
        block_count=len(blocks),
        assets=assets,
    )


def quality_flags(token_count: int, min_tokens: int, max_tokens: int, blocks: list[GovernedBlock]) -> list[str]:
    flags: list[str] = []
    if token_count < min_tokens:
        flags.append("short_chunk")
    elif token_count > max_tokens:
        flags.append("long_chunk")
    else:
        flags.append("length_ok")
    if blocks and all(block.source_anchor for block in blocks):
        flags.append("source_anchor_complete")
    if any(block.type in PROTECTED_BLOCK_TYPES for block in blocks):
        flags.append("protected_block_intact")
    unique_paths = {tuple(block.title_path) for block in blocks if block.title_path}
    unique_top_levels = {block.title_path[0] for block in blocks if block.title_path}
    if len(unique_paths) <= 1:
        flags.append("title_path_consistent")
    elif len(unique_top_levels) > 1:
        flags.append("mixed_title_paths")
    return flags


def derive_tags(blocks: list[GovernedBlock], limit: int = 8) -> list[str]:
    tags: list[str] = []
    for part in common_title_path(blocks):
        tag = normalize_text(part).lower()
        if tag and tag not in tags:
            tags.append(tag)
    type_tags = sorted({block.type for block in blocks})
    for tag in type_tags:
        if tag not in tags:
            tags.append(tag)

    counter: Counter[str] = Counter()
    for block in blocks:
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", block.text.lower()):
            if word not in STOPWORDS:
                counter[word] += 1
    for word, _count in counter.most_common(limit):
        if word not in tags:
            tags.append(word)
        if len(tags) >= limit:
            break
    return tags[:limit]


def derive_entity_tags(blocks: list[GovernedBlock]) -> list[str]:
    tags: list[str] = []
    for block in blocks:
        for tag in block.entity_tags:
            if tag not in tags:
                tags.append(tag)
    return tags


def summarize_text(text: str, max_words: int = 48) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    first_sentence = SENTENCE_SPLIT_RE.split(text, maxsplit=1)[0].strip()
    words = first_sentence.split()
    if len(words) <= max_words:
        return first_sentence
    return " ".join(words[:max_words]) + " ..."


def make_chunk(
    *,
    doc: GovernedDocument,
    strategy: str,
    index: int,
    text: str,
    blocks: list[GovernedBlock],
    min_tokens: int,
    max_tokens: int,
    extra_flags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RagChunk:
    text = normalize_text(text)
    token_count = estimate_tokens(text)
    flags = quality_flags(token_count, min_tokens=min_tokens, max_tokens=max_tokens, blocks=blocks)
    for flag in extra_flags or []:
        if flag not in flags:
            flags.append(flag)
    source_blocks = [block.block_id for block in blocks]
    return RagChunk(
        chunk_id=f"{doc.doc_id}_{strategy}_chunk_{index:05d}",
        doc_id=doc.doc_id,
        dataset=doc.dataset,
        split=doc.split,
        strategy=strategy,  # type: ignore[arg-type]
        text=text,
        token_count=token_count,
        title_path=common_title_path(blocks),
        source_blocks=source_blocks,
        source_anchor=aggregate_source_anchor(doc, blocks),
        tags=derive_tags(blocks),
        summary=summarize_text(text),
        entity_tags=derive_entity_tags(blocks),
        quality_flags=flags,
        metadata=metadata or {},
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_governed_documents(path: str | Path) -> list[GovernedDocument]:
    return [GovernedDocument.model_validate(record) for record in read_jsonl(path)]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
