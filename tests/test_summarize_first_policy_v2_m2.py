"""The TASK-072 M2 results-manifest generator refuses reports it must not summarize."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_summarize_first_policy_v2_m2", ROOT / "scripts" / "summarize_first_policy_v2_m2.py"
)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
POLICY_V1 = json.loads((ROOT / "benchmarks/manifests/apple-policy-v1.json").read_text())


def _report(**changes) -> dict:
    return {
        "mode": "run",
        "protocol": "apple_first_policy_v2_m2",
        "outcome": "M2-FAIL",
        "revision": S.MERGED_REVISION,
        "revision_at_end": S.MERGED_REVISION,
        "tracked_tree_dirty": False,
        "test_split_decoded": False,
        "non_finite_fields": [],
    } | changes


def _summarize(report: dict, sha: str = S.REPORT_SHA256):
    return S.summarize(report, sha, ROOT, POLICY_V1)


def test_refuses_a_foreign_report():
    with pytest.raises(ValueError, match="not the gated run"):
        _summarize(_report(), "0" * 64)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "smoke"}, "not a gated M2 run"),
        ({"mode": "preflight"}, "not a gated M2 run"),
        ({"outcome": "V", "void_reason": "G-hash"}, "void"),
        ({"revision_at_end": "0" * 40}, "merged revision"),
        ({"tracked_tree_dirty": True}, "dirty"),
        ({"test_split_decoded": True}, "test split"),
    ],
)
def test_refuses_a_run_that_is_not_the_clean_gated_run(changes, message):
    with pytest.raises(ValueError, match=message):
        _summarize(_report(**changes))


def test_wilson():
    lo, hi = S.wilson(40, 40)
    assert hi == 1.0 and round(lo, 4) == 0.9124
    lo, hi = S.wilson(12, 40)
    assert (round(lo, 4), round(hi, 4)) == (0.1807, 0.4543)
    assert S.wilson(0, 40)[0] == 0.0
