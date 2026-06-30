from __future__ import annotations

from collections import Counter

from app.chunkers.hsc_rag import HscRagChunker, HscRagConfig
from app.core.schemas import GovernedBlock, GovernedDocument, SourceAnchor


def _anchor(block_id: str, section: str = "Protected Blocks") -> SourceAnchor:
    return SourceAnchor(
        dataset="unit",
        split="test",
        source_doc_id="protected_doc",
        section_name=section,
        extra={"block_id": block_id},
    )


def _block(block_id: str, block_type: str, text: str, order: int) -> GovernedBlock:
    return GovernedBlock(
        block_id=block_id,
        doc_id="protected_doc",
        type=block_type,
        text=text,
        order=order,
        level=1,
        title_path=["Protected Blocks"],
        source_anchor=_anchor(block_id),
    )


def _document_with_protected_blocks() -> GovernedDocument:
    table_text = "\n".join(
        [
            "| metric | value | evidence |",
            "|---|---:|---|",
            "| recall at one | 0.20 | protected table row one |",
            "| recall at three | 0.55 | protected table row two |",
            "| recall at five | 0.70 | protected table row three |",
        ]
    )
    code_text = "\n".join(
        [
            "def score_boundary(buffer_tokens, semantic_distance):",
            "    structure_signal = 1.0",
            "    length_pressure = buffer_tokens / 512",
            "    return 0.45 * structure_signal + 0.35 * semantic_distance + 0.20 * length_pressure",
        ]
    )
    formula_text = (
        "boundary_score = structure_signal times zero point four five plus "
        "semantic_distance times zero point three five plus length_pressure times zero point two"
    )
    return GovernedDocument(
        doc_id="protected_doc",
        dataset="unit",
        split="test",
        source_doc_id="protected_doc",
        title="Protected Block Test",
        normalization_status="simulated_governed",
        blocks=[
            _block("p_intro", "paragraph", "Introductory governed text before protected content.", 1),
            _block("tbl_001", "table", table_text, 2),
            _block("code_001", "code", code_text, 3),
            _block("formula_001", "formula", formula_text, 4),
            _block("p_outro", "paragraph", "Closing governed text after protected content.", 5),
        ],
    )


def test_hsc_rag_keeps_table_code_and_formula_blocks_intact():
    doc = _document_with_protected_blocks()
    chunker = HscRagChunker(HscRagConfig(min_tokens=4, target_tokens=8, max_tokens=12))

    chunks = chunker.chunk_document(doc)
    membership = Counter(
        block_id
        for chunk in chunks
        for block_id in chunk.source_blocks
        if block_id in {"tbl_001", "code_001", "formula_001"}
    )

    assert membership == {"tbl_001": 1, "code_001": 1, "formula_001": 1}

    for block_id in ["tbl_001", "code_001", "formula_001"]:
        containing_chunks = [chunk for chunk in chunks if block_id in chunk.source_blocks]
        assert len(containing_chunks) == 1
        chunk = containing_chunks[0]
        assert chunk.source_blocks == [block_id]
        assert "protected_block_intact" in chunk.quality_flags
        assert "kept_protected_block_intact" in chunk.quality_flags
        assert chunk.source_anchor.first_block_id == block_id
        assert chunk.source_anchor.last_block_id == block_id
        assert chunk.source_anchor.block_count == 1
