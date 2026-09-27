"""TASK-066 summarize script: the rows that also match follow protocol section 11 (synthetic)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_summarize_token_dynamics", ROOT / "scripts" / "summarize_token_dynamics.py"
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _report(seeds, ceiling=True):
    return {
        "gates": {
            str(i): {"seed": {"passes": p, "dynamics": d, "latent": lt}}
            for i, (p, d, lt) in enumerate(seeds)
        },
        "ceiling": {"8": {"meets_t1": True}, "16": {"meets_t1": ceiling}},
    }


def test_no_dynamics_is_otherwise_only():
    apple_lost = _report([(False, True, True)] * 3)
    assert S.rows_also_matching(apple_lost) == ["WM-TOK-APPLE-LOST", "WM-TOK-COLLAPSE"]
    collapse = _report([(False, False, True)] * 3)
    assert S.rows_also_matching(collapse) == ["WM-TOK-COLLAPSE"]
    nothing = _report([(False, False, False)] * 3)
    assert S.rows_also_matching(nothing) == ["WM-TOK-NO-DYNAMICS"]


def test_pass_unstable_and_ceiling():
    assert S.rows_also_matching(_report([(True, True, True)] * 3)) == ["WM-TOK-DYNAMICS"]
    unstable = _report([(True, True, True), (False, True, True), (False, True, True)])
    assert S.rows_also_matching(unstable) == ["WM-TOK-UNSTABLE"]
    ceiling = _report([(False, True, True)] * 3, ceiling=False)
    assert S.rows_also_matching(ceiling)[0] == "WM-TOK-CEILING"
