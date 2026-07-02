from __future__ import annotations

from app.chunkers.hsc_rag import HscRagChunker, HscRagConfig
from app.core.schemas import GovernedBlock, GovernedDocument, SourceAnchor


def _anchor(block_id: str, section: str) -> SourceAnchor:
    return SourceAnchor(
        dataset="unit",
        split="test",
        source_doc_id="adaptive_doc",
        section_name=section,
        extra={"block_id": block_id},
    )


def _block(
    *,
    index: int,
    title_path: list[str],
    text: str,
    block_type: str = "paragraph",
) -> GovernedBlock:
    block_id = f"b{index:03d}"
    section = " / ".join(title_path)
    return GovernedBlock(
        block_id=block_id,
        doc_id="adaptive_doc",
        type=block_type,
        text=text,
        order=index,
        level=len(title_path),
        title_path=title_path,
        source_anchor=_anchor(block_id, section),
    )


def _document(blocks: list[GovernedBlock]) -> GovernedDocument:
    return GovernedDocument(
        doc_id="adaptive_doc",
        dataset="unit",
        split="test",
        source_doc_id="adaptive_doc",
        title="Adaptive Boundary Unit Test",
        normalization_status="simulated_governed",
        blocks=blocks,
    )


def _first_adaptive_metadata(doc: GovernedDocument, config: HscRagConfig) -> dict:
    chunks = HscRagChunker(config).chunk_document(doc)
    assert chunks
    metadata = chunks[0].metadata.get("adaptive_boundary")
    assert isinstance(metadata, dict)
    return metadata


def test_adaptive_boundary_relaxes_for_many_short_cohesive_blocks():
    blocks = [
        _block(
            index=index,
            title_path=["Shared Section"],
            text="alpha beta gamma delta shared context",
        )
        for index in range(1, 31)
    ]
    metadata = _first_adaptive_metadata(
        _document(blocks),
        HscRagConfig(min_tokens=20, target_tokens=60, max_tokens=120),
    )

    assert metadata["method"] == "document_statistics"
    assert metadata["profile"] == "high_context_coverage"
    assert metadata["boundary_strength"] == "loose"
    assert metadata["decision_basis"]["uses_dataset_name"] is False
    assert metadata["decision_basis"]["uses_gold_evidence"] is False
    assert metadata["decision_basis"]["uses_query_text"] is False
    assert metadata["effective_config"]["target_tokens"] > metadata["base_config"]["target_tokens"]
    assert (
        metadata["effective_config"]["semantic_boundary_threshold"]
        > metadata["base_config"]["semantic_boundary_threshold"]
    )


def test_adaptive_boundary_preserves_structure_when_titles_shift_frequently():
    blocks = [
        _block(
            index=index,
            title_path=[f"Section {index:02d}"],
            text=f"topic{index:02d} unique{index:02d} evidence{index:02d} signal{index:02d}",
        )
        for index in range(1, 13)
    ]
    metadata = _first_adaptive_metadata(
        _document(blocks),
        HscRagConfig(min_tokens=20, target_tokens=60, max_tokens=120),
    )

    assert metadata["profile"] == "structure_preserving"
    assert metadata["boundary_strength"] == "strong"
    assert metadata["stats"]["top_title_transition_rate"] > 0.8
    assert (
        metadata["effective_config"]["semantic_boundary_threshold"]
        <= metadata["base_config"]["semantic_boundary_threshold"]
    )
    assert metadata["effective_config"]["target_tokens"] == metadata["base_config"]["target_tokens"]


def test_adaptive_boundary_can_be_disabled_for_fixed_threshold_experiments():
    blocks = [
        _block(
            index=index,
            title_path=["Shared Section"],
            text="alpha beta gamma delta shared context",
        )
        for index in range(1, 8)
    ]
    chunks = HscRagChunker(
        HscRagConfig(
            min_tokens=20,
            target_tokens=60,
            max_tokens=120,
            adaptive_boundary=False,
        )
    ).chunk_document(_document(blocks))

    assert chunks
    for chunk in chunks:
        assert "adaptive_boundary" not in chunk.metadata
        assert "hsc_adaptive_boundary" not in chunk.quality_flags
