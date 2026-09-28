"""The TASK-072 results-manifest generator refuses reports it must not summarize."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_summarize_first_policy_v2_linux", ROOT / "scripts" / "summarize_first_policy_v2_linux.py"
)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _minimal_report(**stages) -> dict:
    return {
        "protocol": "apple_first_policy_v2_linux",
        "smoke": False,
        "data": {"seeds": {"D": list(range(52000, 52016))}},
        "stages": stages,
    }


def _summarize(report: dict) -> dict:
    return S.summarize(
        report,
        {},
        report_sha256="0" * 64,
        run1_sha256=S.RUN1_REPORT_SHA256,
        run_root=ROOT,
        log=ROOT / "README.md",
    )


def test_band_scan_finds_a_cohort_c_seed_anywhere():
    report = _minimal_report(M1={"attempts": {"P-3": [{"seed": 52000}, {"seed": 45310}]}})
    assert S.band_integers(report) == [45310]
    assert S.band_integers({"a": [True, 44999, 46001, 45000.5, "45310"]}) == []
    assert S.band_integers({"edge": [45000, 46000]}) == [46000, 45000]


def test_summarize_refuses_a_report_with_a_cohort_c_seed():
    report = _minimal_report(corpus={"attempts": [{"seed": 45300}]})
    with pytest.raises(ValueError, match="cohort-C band"):
        _summarize(report)


def test_summarize_refuses_a_foreign_run1_report_and_a_smoke():
    with pytest.raises(ValueError, match="run-1 report sha256"):
        S.summarize(
            _minimal_report(),
            {},
            report_sha256="0" * 64,
            run1_sha256="f" * 64,
            run_root=ROOT,
            log=ROOT / "README.md",
        )
    with pytest.raises(ValueError, match="smoke"):
        _summarize({**_minimal_report(), "smoke": True})


def test_void_is_derived_from_the_report():
    assert S.is_void({"outcome": "V", "replication": {"row": "V"}})
    assert S.is_void({"outcome": "M1-PASS", "replication": {"row": "V"}})
    assert not S.is_void({"outcome": "M1-PASS", "replication": {"row": "REPLICATED"}})


def test_statistics():
    assert S.mcnemar(4, 0) == 0.125
    assert S.mcnemar(9, 0) == pytest.approx(0.00390625)
    assert S.mcnemar(0, 0) == 1.0
    lo, hi = S.wilson(2, 16)
    assert round(lo, 2) == 0.03 and round(hi, 2) == 0.36
    assert S.wilson(16, 16)[1] == 1.0 and S.wilson(0, 16)[0] == 0.0
