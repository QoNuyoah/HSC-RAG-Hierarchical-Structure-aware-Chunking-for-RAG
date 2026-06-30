from __future__ import annotations

from app.core.schemas import ChunkAgentRequest, ChunkAgentResponse
from app.services.chunking_service import run_chunk_request


def test_topic11_example_response_is_stable(
    topic11_request_payload,
    topic11_response_payload,
):
    request = ChunkAgentRequest.model_validate(topic11_request_payload)
    expected = ChunkAgentResponse.model_validate(topic11_response_payload).model_dump(mode="json")

    actual = run_chunk_request(request).model_dump(mode="json")

    assert actual == expected
    assert actual["report"]["input_contract"] == "GovernedDocument"
    assert actual["report"]["output_contract"] == "RagChunk[]"
    assert actual["report"]["governance_stage"] == "post_normalization_packaging"
