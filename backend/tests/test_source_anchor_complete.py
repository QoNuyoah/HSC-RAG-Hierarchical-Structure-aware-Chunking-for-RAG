from __future__ import annotations

from app.core.schemas import ChunkAgentRequest
from app.services.chunking_service import run_chunk_request


def test_chunks_keep_complete_source_anchor_contract(topic11_request_payload):
    request = ChunkAgentRequest.model_validate(topic11_request_payload)
    response = run_chunk_request(request)
    source_block_ids = {block.block_id for block in request.document.blocks}

    assert response.chunks
    for chunk in response.chunks:
        assert chunk.source_blocks
        assert set(chunk.source_blocks).issubset(source_block_ids)
        assert "source_anchor_complete" in chunk.quality_flags

        anchor = chunk.source_anchor
        assert anchor.dataset == chunk.dataset
        assert anchor.split == chunk.split
        assert anchor.source_doc_id == request.document.source_doc_id
        assert anchor.first_block_id == chunk.source_blocks[0]
        assert anchor.last_block_id == chunk.source_blocks[-1]
        assert anchor.block_count == len(chunk.source_blocks)
