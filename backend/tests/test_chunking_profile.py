from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chunking import apply_chunking_profile, run_strategy  # noqa: E402


def _base_args(**overrides):
    args = Namespace(
        chunking_profile="default",
        effective_chunking_profile="default",
        chunking_profile_overrides={},
        fixed_target=512,
        fixed_overlap=64,
        recursive_target=512,
        recursive_overlap=64,
        semantic_min=160,
        semantic_target=512,
        semantic_max=768,
        semantic_breakpoint_percentile=75.0,
        hsc_min=180,
        hsc_target=512,
        hsc_max=900,
        hsc_boundary_threshold=0.62,
        hsc_soft_boundary_threshold=0.52,
        hsc_semantic_distance_threshold=0.72,
        hsc_semantic_window_blocks=3,
        hsc_adaptive_boundary=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_zh_legal_chunking_profile_sets_hsc_min_length_prior():
    args = apply_chunking_profile(
        _base_args(chunking_profile="zh_legal"),
        argv=["--chunking-profile", "zh_legal"],
    )

    assert args.hsc_min == 128
    assert args.effective_chunking_profile == "zh_legal"
    assert args.chunking_profile_overrides == {"hsc_min": 128}


def test_zh_legal_chunking_profile_does_not_override_explicit_hsc_min():
    args = apply_chunking_profile(
        _base_args(chunking_profile="zh_legal", hsc_min=160),
        argv=["--chunking-profile", "zh_legal", "--hsc-min", "160"],
    )

    assert args.hsc_min == 160
    assert args.effective_chunking_profile == "zh_legal"
    assert args.chunking_profile_overrides == {}


def test_zh_legal_chunking_profile_keeps_adaptive_boundary_enabled():
    args = apply_chunking_profile(
        _base_args(chunking_profile="zh_legal"),
        argv=["--chunking-profile", "zh_legal"],
    )

    _, config = run_strategy("hsc_rag", [], args)

    assert config["min_tokens"] == 128
    assert config["adaptive_boundary"] is True
    assert config["chunking_profile"] == "zh_legal"
