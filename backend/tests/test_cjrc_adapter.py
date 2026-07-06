from __future__ import annotations

import json

from app.adapters.cjrc_adapter import CjrcAdapter


def test_cjrc_adapter_maps_case_span_to_gold_block(tmp_path):
    context = "经审理查明:原告与被告因夫妻感情破裂离婚。被告每月支付生活费800元。"
    answer = "生活费800元"
    payload = {
        "data": [
            {
                "caseid": "case_001",
                "domain": "civil",
                "paragraphs": [
                    {
                        "casename": "变更抚养关系纠纷",
                        "context": context,
                        "qas": [
                            {
                                "id": "case_001_q1",
                                "question": "被告每月支付什么？",
                                "answers": [
                                    {
                                        "text": answer,
                                        "answer_start": context.index(answer),
                                    }
                                ],
                                "is_impossible": "false",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    input_path = tmp_path / "dev_ground_truth.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    adapter = CjrcAdapter(input_path, block_mode="sentence")
    docs, evidence_records, stats = adapter.convert(split="dev")

    assert stats.documents == 1
    assert stats.queries == 1
    assert stats.matched_evidence_items == 1
    assert docs[0].dataset == "cjrc_cail2019"
    assert docs[0].source_doc_id == "case_001"
    assert docs[0].title == "变更抚养关系纠纷"
    assert docs[0].metadata["domain"] == "civil"

    query = docs[0].queries[0]
    evidence = evidence_records[0]

    assert query.gold_block_ids
    assert evidence.gold_block_ids == query.gold_block_ids
    assert any(answer in text for text in query.gold_evidence_texts)
    assert evidence.evidence_matches[0]["method"] == "answer_start_overlap"
    assert evidence.evidence_matches[0]["block_id"] == query.gold_block_ids[0]

    gold_block = next(block for block in docs[0].blocks if block.block_id == query.gold_block_ids[0])
    assert gold_block.source_anchor.source_doc_id == "case_001"
    assert gold_block.source_anchor.extra["domain"] == "civil"
    assert gold_block.source_anchor.extra["casename"] == "变更抚养关系纠纷"


def test_cjrc_adapter_keeps_spanless_yes_no_out_of_gold_eval(tmp_path):
    context = "经审理查明:原告提交了合同。"
    payload = {
        "data": [
            {
                "caseid": "case_002",
                "domain": "civil",
                "paragraphs": [
                    {
                        "casename": "合同纠纷",
                        "context": context,
                        "qas": [
                            {
                                "id": "case_002_q1",
                                "question": "原告是否提交合同？",
                                "answers": [{"text": "YES", "answer_start": -1}],
                                "is_impossible": "false",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    input_path = tmp_path / "dev_ground_truth.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    adapter = CjrcAdapter(input_path, block_mode="sentence")
    docs, evidence_records, stats = adapter.convert(split="dev")

    assert stats.spanless_answerable_queries == 1
    assert docs[0].queries[0].answer_type == "yes_no_without_span"
    assert docs[0].queries[0].is_unanswerable is True
    assert evidence_records[0].gold_block_ids == []
