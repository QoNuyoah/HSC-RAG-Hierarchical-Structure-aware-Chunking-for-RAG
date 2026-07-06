from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_longbench_mcq_eval import lexical_judge, normalize_answer_label  # noqa: E402


def test_lexical_judge_selects_choice_supported_by_retrieved_context():
    result = lexical_judge(
        question="How should smart court construction be promoted?",
        choices={
            "A": "Change office decorations.",
            "B": "Reduce public services.",
            "C": "Replace all courts.",
            "D": "Use advanced information systems to improve case handling.",
        },
        retrieved_text=(
            "The report says smart court construction should use advanced "
            "information systems to improve case handling and judicial efficiency."
        ),
    )

    assert result["answer"] == "D"
    assert result["provider_execution"] == "lexical_offline_judge"


def test_normalize_answer_label_accepts_labeled_answer_text():
    assert normalize_answer_label("D: Use advanced information systems") == "D"
