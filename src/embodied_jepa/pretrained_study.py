"""TASK-063 decision code (``apple_pretrained_encoder_v1``): arms, floors, Holm, rows, guards.

The probe is TASK-061's L probe run by ``info_ceiling`` / ``observation_reprobe`` unchanged, with
TASK-062's ``encoder_study.arm_pvalues`` and ``qualifier``. This module adds only what the
protocol (§5-§11) changes: two decisional arms (P-cls, P-tok) with their same-architecture
random-init floors (R-cls, R-tok), Holm over the two, the arm states (no collapse gate: nothing is
trained), the first-matching-row decision, and the guards G-weights, G-frames, G-repro and a
G-anchor over two anchor sets (TASK-061 and TASK-062 reports).

NumPy only; torch is never imported here.
"""

from __future__ import annotations

from embodied_jepa import encoder_study as es
from embodied_jepa import info_ceiling as ic
from embodied_jepa import observation_reprobe as orp
from embodied_jepa.contracts import ContractError

GuardError = ic.GuardError

ARMS = ("P-cls", "P-tok")  # listed order = Holm tie-break order
ARM_FLOOR = {"P-cls": "R-cls", "P-tok": "R-tok"}
ARM_POINT = {"P-cls": "cls", "P-tok": "tokens"}
SOURCE = {"P-cls": "P_cls", "P-tok": "P_tok", "R-cls": "R_cls", "R-tok": "R_tok"}
ALPHA_ONE_SIDED = 0.025
PER_ARM_SECONDS = 1800.0
GLOBAL_WALL_SECONDS = 7200.0
DEVICE = "cpu"
ANCHOR_SETS = {
    "task061": ("L_raw", "L_E0", "L_random"),
    "task062": ("F_tok", "E0_tok"),
}
ANCHOR_TOLERANCE = es.ANCHOR_TOLERANCE
ROWS = ("V", "O-PT-POOLED", "O-PT-TOKENS", "O-PT-INCOMPLETE", "O-PT-FLOOR", "O-PT-NONE")


# ----- multiplicity, states, pass rule ----------------------------------------------------------
def holm(pvalues: dict) -> dict:
    return orp.holm(pvalues, alpha=ALPHA_ONE_SIDED, order=ARMS)


def arm_state(*, evaluated: bool) -> str:
    return "evaluated" if evaluated else "not evaluated"


def over_cap(seconds: float, cap: float = PER_ARM_SECONDS) -> bool:
    return bool(seconds > cap)


def passes(*, evaluated: bool, succeeds: bool, beats_floor: bool, rejected: bool, spurious: bool):
    return bool(evaluated and succeeds and beats_floor and rejected and not spurious)


def qualifier(*, passes: bool, succeeds: bool, beats_floor: bool, rejected: bool, beats_prior):
    return es.qualifier(
        passes=passes,
        succeeds=succeeds,
        beats_floor=beats_floor,
        rejected=rejected,
        beats_prior=beats_prior,
    )


def decide(*, void: bool, arms: dict) -> dict:
    """First matching row (§11). ``arms``: arm -> {"passes", "state", "succeeds", "spurious"}."""
    missing = set(ARMS) - set(arms)
    if missing:
        raise ContractError(f"decision needs every arm; missing {missing}")
    if void:
        return {"outcome": "V"}
    passed = {a: bool(arms[a]["passes"]) for a in ARMS}
    if passed["P-cls"]:
        return {"outcome": "O-PT-POOLED", "passed": passed}
    if passed["P-tok"]:
        return {"outcome": "O-PT-TOKENS", "passed": passed}
    unevaluated = [a for a in ARMS if arms[a]["state"] == "not evaluated"]
    if unevaluated:
        return {"outcome": "O-PT-INCOMPLETE", "not_evaluated": unevaluated, "passed": passed}
    exposed = [a for a in ARMS if arms[a]["succeeds"] and not arms[a]["spurious"]]
    if exposed:
        return {"outcome": "O-PT-FLOOR", "succeeding_arms": exposed, "passed": passed}
    return {"outcome": "O-PT-NONE", "passed": passed}


def abandonment_fires(outcome: str) -> bool:
    return outcome in ("O-PT-FLOOR", "O-PT-NONE")


# ----- guards ---------------------------------------------------------------------------------
def check_weights(pretrained_digest: str, floor_digest: str, manifest: dict) -> None:
    """G-weights (digest part; the file hashes are checked by ``pretrained_encoder``)."""
    want_p = manifest["encoder"]["pretrained_weights_digest"]
    want_f = manifest["floors"]["weights_digest"]
    if pretrained_digest != want_p:
        raise GuardError(f"G-weights: pretrained digest {pretrained_digest[:12]} != pinned")
    if floor_digest != want_f:
        raise GuardError(f"G-weights: floor digest {floor_digest[:12]} != the seed-0 init")


def check_frames(got_sha256: str, want_sha256: str) -> None:
    if got_sha256 != want_sha256:
        raise GuardError(f"G-frames: post-look frames {got_sha256[:12]} != {want_sha256[:12]}")


def check_repro(first: dict, second: dict) -> None:
    import numpy as np

    for key in first:
        if key not in second or not np.array_equal(first[key], second[key]):
            raise GuardError(f"G-repro: second forward pass differs at {key}")


def anchor_compare(reference: dict, recomputed: dict, sources) -> dict:
    """TASK-062's G-anchor verdict (``encoder_study.anchor_compare``) over ``sources``."""
    diffs = []
    for source in sources:
        if source not in reference or source not in recomputed:
            raise GuardError(f"G-anchor: missing anchor source {source}")
        es._walk(es._jsonable(reference[source]), es._jsonable(recomputed[source]), source, diffs)
    numeric = [d for d in diffs if d[1] == "numeric"]
    discrete = [d for d in diffs if d[1] == "discrete"]
    selection_like = [d for d in numeric if any(k in d[0] for k in es._DISCRETE_NUMERIC)]
    max_delta = max((abs(d[2] - d[3]) for d in numeric), default=0.0)
    if discrete or selection_like or max_delta > ANCHOR_TOLERANCE:
        verdict = "void"
    elif numeric:
        verdict = "caveat"
    else:
        verdict = "exact"
    return {
        "verdict": verdict,
        "sources": list(sources),
        "max_abs_delta": max_delta,
        "numeric_differences": len(numeric),
        "discrete_differences": len(discrete) + len(selection_like),
        "examples": [list(map(str, d)) for d in (discrete + selection_like + numeric)[:20]],
    }


def check_anchor(comparison: dict, label: str) -> None:
    if comparison["verdict"] == "void":
        raise GuardError(
            f"G-anchor ({label}): the anchors did not reproduce "
            f"(max |delta| {comparison['max_abs_delta']}, "
            f"{comparison['discrete_differences']} discrete differences)"
        )
