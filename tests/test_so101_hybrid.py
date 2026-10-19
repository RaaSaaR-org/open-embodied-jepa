"""TASK-102 (SO-101 S5, second study) Stage 0 tests: cohorts, salts, the hand-over rule, the τ_h
rule, the window rule and the window builder (on a synthetic corpus), the configuration rule,
the bar rule, every row of §7.7 and §9 in both directions, the ledger, the guards, the
follower-from-hand-over path (G-zero), S+G's approach, the truth sentinel around G_loc, and the
byte-identity of every pinned file. CPU only; physics tests skip only when MuJoCo or the pinned
SO-101 assets are absent, and the synthetic-corpus tests only when pyarrow is absent."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid as hy
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
SO101_MANIFEST = ROOT / "assets/so101_manifest.json"


def _fetched() -> bool:
    manifest = json.loads(SO101_MANIFEST.read_text())
    root = ROOT / manifest["local_path"]
    return all((root / relative).is_file() for relative in manifest["sha256"])


# ----- cohorts, seeds, salts --
def test_cohorts_are_the_declared_ranges_and_disjoint():
    want = {
        "P": (103000, 103031, 103032, 103047),
        "Pp": (103060, 103091, 103092, 103099),
        "D": (103100, 103163, 103164, 103199),
        "S": (103200, 103327, 103400, 103463),
        "A": (103500, 103531, 103532, 103547),
        "Dp": (103600, 103663, 103664, 103699),
        "Ap": (103700, 103731, 103732, 103747),
    }
    seen = set()
    for name, (lo, hi, rlo, rhi) in want.items():
        c = hy.COHORTS[name]
        assert c.seeds == tuple(range(lo, hi + 1))
        assert c.reserve_seeds == tuple(range(rlo, rhi + 1))
        both = set(c.seeds) | set(c.reserve_seeds)
        assert not both & seen
        seen |= both
    debug = set(hy.COHORTS["debug"].seeds)
    assert debug == set(range(103900, 104000)) and not debug & seen
    # disjoint from every TASK-101 range (all retired, R28.93)
    for c in cl.COHORTS.values():
        assert not (set(c.seeds) | set(c.reserve_seeds)) & (seen | debug)


def test_seeds_outside_a_cohort_are_refused():
    assert hy.check_seed("P", 103047) == 103047
    for name, seed in (("P", 103048), ("A", 103100), ("D", 103200), ("S", 103399)):
        with pytest.raises(hy.GuardError, match="G-seed"):
            hy.check_seed(name, seed)
    with pytest.raises(hy.GuardError):
        hy.check_seed("P", 103000.0)


def test_salts_and_derived_streams():
    assert hy.SALTS == {
        "cem": 10110,
        "rand": 10111,
        "offset": 10112,
        "bootstrap": 10113,
        "power": 10114,
        "gloc": 10115,
        "marker": 10107,
    }
    assert hy.planner_seed(103100) == 10110 * 10**6 + 103100
    a = hy.rand_rng(103100).uniform(size=3)
    assert np.array_equal(a, np.random.default_rng([10111, 103100]).uniform(size=3))
    g = hy.gloc_rng(2).integers(0, 100, 5)
    assert np.array_equal(g, np.random.default_rng([10115, 2]).integers(0, 100, 5))


def test_p_offsets_have_the_declared_rms_and_one_direction_per_seed():
    u = hy.p_direction(103000)
    n = np.random.default_rng([10112, 103000]).standard_normal(5)
    assert np.allclose(u, n / math.sqrt(np.mean(n * n)), rtol=0, atol=0)
    for r in hy.P_LEVELS_RAD:
        d = hy.p_offset(103000, r)
        assert math.isclose(math.sqrt(np.mean(d * d)), r, abs_tol=1e-12)
        if r:
            assert np.allclose(d / r, u)
    with pytest.raises(ContractError):
        hy.p_offset(103000, 0.06)


# ----- the hand-over rule (§5.2) --
def _goal():
    return np.array([0.1, -0.2, 0.3, 1.2, -0.4, 1.5])


def test_handover_needs_three_consecutive_near_commands():
    g = _goal()
    rule = hy.HandoverRule(g, 0.08)
    near = g.copy()
    near[:5] += 0.07  # RMS 0.07 <= 0.08
    far = g.copy()
    far[:5] += 0.09
    jaw_off = near.copy()
    jaw_off[5] += 0.16
    seq = [near, near, far, near, jaw_off, near, near, near]
    phases = [rule.update(q) for q in seq]
    assert phases == [1, 1, 1, 1, 1, 1, 1, 2]
    assert rule.event["how"] == "tolerance" and rule.event["command"] == 7
    assert rule.event["phase1_commands"] == 7
    assert math.isclose(rule.event["arm_rms_rad"], 0.07, rel_tol=1e-9)
    assert rule.update(far) == 2  # phase 2 is kept to the end
    assert math.isclose(rule.closest, 0.07, rel_tol=1e-9)


def test_handover_times_out_after_60_phase1_commands():
    g = _goal()
    rule = hy.HandoverRule(g, 0.04)
    far = g.copy()
    far[:5] += 0.5
    phases = [rule.update(far) for _ in range(65)]
    assert phases[:60] == [1] * 60 and phases[60:] == [2] * 5
    assert rule.event == {
        "command": 60,
        "how": "timeout",
        "phase1_commands": 60,
        "arm_rms_rad": pytest.approx(0.5),
        "jaw_error_rad": 0.0,
    }


def test_handover_tau_is_capped():
    with pytest.raises(ContractError):
        hy.HandoverRule(_goal(), 0.16)
    with pytest.raises(ContractError):
        hy.HandoverRule(_goal()[:5], 0.04)


# ----- τ_h (§5.2) --
def test_tau_env_rule_on_the_recorded_envelope_and_synthetic_columns():
    recorded = dict.fromkeys(hy.ENVELOPE_LEVELS_RAD, 32)  # TASK-101 A2's "pre" column
    assert hy.tau_env_rule(recorded) == {"tau_env": 0.16, "tau_max": 0.12}
    broken = recorded | {0.12: 29}
    assert hy.tau_env_rule(broken) == {"tau_env": 0.08, "tau_max": 0.06}
    assert hy.tau_env_rule(recorded | {0.02: 29})["tau_max"] is None
    assert hy.tau_env_rule(recorded | {0.04: 29})["tau_max"] is None  # τ_env 0.02: no margin
    with pytest.raises(ContractError):
        hy.tau_env_rule({0.02: 32})


@pytest.mark.parametrize(
    ("k", "tau", "fallback"),
    [
        ({0.0: 30, 0.04: 30, 0.08: 28, 0.12: 27}, 0.12, False),
        ({0.0: 30, 0.04: 27, 0.08: 26, 0.12: 30}, 0.04, False),
        ({0.0: 30, 0.04: 28, 0.08: 26, 0.12: 30}, 0.04, False),  # 0.08 breaks the run
        ({0.0: 30, 0.04: 26, 0.08: 30, 0.12: 30}, 0.04, True),  # nothing qualifies
        ({0.0: 20, 0.04: 17, 0.08: 17, 0.12: 16}, 0.08, False),
        ({0.0: 0, 0.04: 0, 0.08: 0, 0.12: 0}, 0.12, False),
    ],
)
def test_tau_h_rule_every_branch(k, tau, fallback):
    out = hy.tau_h_rule(k)
    assert out["tau_h"] == tau and out["fallback"] is fallback
    assert out["tau_h"] <= hy.TAU_MAX
    assert out["gloc_weak"] is (k[0.0] < 16)


def test_tau_h_rule_never_exceeds_tau_max_and_needs_every_level():
    k = {0.0: 30, 0.04: 30, 0.08: 30, 0.12: 30}
    assert hy.tau_h_rule(k, tau_max=0.08)["tau_h"] == 0.08
    with pytest.raises(ContractError):
        hy.tau_h_rule({0.0: 30, 0.04: 30})
    with pytest.raises(ContractError):
        hy.tau_h_rule(k | {0.04: 33})


# ----- G_loc's inputs (§5.3) --
def test_clock_and_nonlatent_inputs():
    c = hy.clock(np.array([0, 4, 6]))
    assert c.shape == (3, 17) and hy.CLOCK_PERIODS == (4, 8, 16, 32, 64, 128, 256, 512)
    assert np.allclose(c[:, 0], [0.0, 0.04, 0.06])
    assert np.allclose(c[1, 1], 0.0, atol=1e-12) and np.allclose(c[1, 9], 1.0)  # sin, cos at 4
    x = hy.nonlatent(np.ones(6), 2 * np.ones(6), 3 * np.ones(6), 5)
    assert x.dtype == np.float32 and x.shape == (35,)
    assert np.array_equal(x[:18], np.repeat([1.0, 2.0, 3.0], 6).astype(np.float32))
    assert hy.INPUT_DIM == 12_323 and hy.NONLATENT_DIM == 35
    with pytest.raises(hy.GuardError):
        hy.nonlatent(np.full(6, np.nan), np.ones(6), np.ones(6), 0)


# ----- the window rule (§5.4) --
D_, C_, L_, H_, R_, T_ = 2, 3, 4, 5, 1, 6  # descend, close, lift, hold, reach, transport labels


def _phase(spec):
    out = []
    for label, n in spec:
        out += [label] * n
    return np.asarray(out + [0], np.int8)  # the final frame's label


def test_window_rule_on_synthetic_attempts():
    # attempt 0 fails (not grasped), the regrasp grasps and is followed by 50 holds
    phase = _phase(
        [(R_, 5), (D_, 6), (C_, 3), (L_, 4), (R_, 3), (R_, 4), (D_, 5), (C_, 2), (L_, 3), (H_, 50)]
    )
    n = len(phase) - 1
    attempts = [
        {"failure": "none", "grasped": False, "lift_end_step": 18},
        {"failure": "none", "grasped": True, "lift_end_step": 35},
        {"failure": "early_close", "grasped": True, "lift_end_step": 35},
        {"failure": "none", "grasped": None, "lift_end_step": None},
    ]
    out = hy.grasp_windows(phase, attempts, n, descend=D_, hold=H_)
    assert out["excluded"] == {
        "not_grasped": 1,
        "early_close": 1,
        "no_lift_end": 1,
        "no_descend": 0,
    }
    (w,) = out["windows"]
    assert w["start"] == 25 and w["lift_end"] == 35 and w["holds"] == 40 and w["end"] == 75
    # the last contiguous descend run only: an earlier, interrupted descent is not included
    phase2 = _phase([(D_, 3), (C_, 2), (D_, 4), (C_, 2), (L_, 3), (T_, 5)])
    out2 = hy.grasp_windows(
        phase2,
        [{"failure": "none", "grasped": True, "lift_end_step": 14}],
        len(phase2) - 1,
        descend=D_,
        hold=H_,
    )
    assert out2["windows"][0]["start"] == 5 and out2["windows"][0]["end"] == 14
    assert out2["windows"][0]["holds"] == 0
    # holds stop at the episode's end
    phase3 = _phase([(D_, 2), (C_, 1), (L_, 1), (H_, 3)])
    w3 = hy.grasp_windows(
        phase3, [{"failure": "none", "grasped": True, "lift_end_step": 4}], 7, descend=D_, hold=H_
    )["windows"][0]
    assert w3["end"] == 7 and w3["holds"] == 3


def test_window_rule_rejects_impossible_lift_ends_and_counts_missing_descents():
    phase = _phase([(C_, 3), (L_, 2)])
    out = hy.grasp_windows(
        phase, [{"failure": "none", "grasped": True, "lift_end_step": 5}], 5, descend=D_, hold=H_
    )
    assert out["excluded"]["no_descend"] == 1 and not out["windows"]
    with pytest.raises(ContractError):
        hy.grasp_windows(
            phase,
            [{"failure": "none", "grasped": True, "lift_end_step": 9}],
            5,
            descend=D_,
            hold=H_,
        )


def test_window_task_success_needs_twenty_steps():
    rise = np.r_[np.zeros(5), np.full(20, 0.06), np.zeros(5)]
    contact = np.ones(30, bool)
    assert hy.window_task_success(rise, contact, 0, 29)
    contact[10] = False
    assert not hy.window_task_success(rise, contact, 0, 29)


def test_the_test_split_is_refused():
    with pytest.raises(hy.GuardError, match="G-test"):
        hy.check_split("test")
    from embodied_jepa import so101_hybrid_runtime as hr

    with pytest.raises(hy.GuardError, match="G-test"):  # before any file is opened
        hr.build_split("test", "/nonexistent", "/nonexistent", "/nonexistent")
    with pytest.raises(hy.GuardError, match="G-test"):
        hr.feature_check("/nonexistent", ("train", "test"))


# ----- the window builder on a synthetic corpus --
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def synthetic(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from embodied_jepa.so101_play_corpus import PHASES, REST_Z

    d, c, li, h = (PHASES.index(x) for x in ("descend", "close", "lift", "hold"))
    corpus, features = tmp_path / "corpus", tmp_path / "features"
    shard = corpus / "shard-00"
    (shard / "meta").mkdir(parents=True)
    (shard / "data").mkdir()
    (shard / "play").mkdir()
    features.mkdir()
    rng = np.random.default_rng(0)
    table, rows, sha = [], [], {}
    actions_all, phase_all, lat_all = [], [], []
    offset = 0
    eids = {"train": ["e-1", "e-2"], "val": ["e-3"], "test": ["e-4"]}
    for k, eid in enumerate(["e-1", "e-2", "e-3"]):
        phase = np.asarray([1] * 5 + [d] * 6 + [c] * 3 + [li] * 4 + [h] * 7 + [0], np.int8)
        n = len(phase)
        state = rng.normal(size=(n, 12)).astype(np.float32)
        action = np.zeros((n, 6), np.float32)
        action[:-1] = rng.uniform(-1, 1, (n - 1, 6))
        rel = f"data/file-{k}.parquet"
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array(state.tolist(), pa.list_(pa.float32())),
                    "action": pa.array(action.tolist(), pa.list_(pa.float32())),
                }
            ),
            shard / rel,
        )
        sha[rel] = _sha(shard / rel)
        cube = np.zeros((n, 3))
        cube[:, 2] = REST_Z
        side = shard / "play" / f"{eid}.npz"
        np.savez(
            side,
            phase=phase,
            targets=rng.normal(size=(n, 6)).astype(np.float32),
            cube_position=cube.astype(np.float32),
            gripper_contact=np.zeros(n, bool),
        )
        attempts = [{"failure": "none", "grasped": True, "lift_end_step": 18}]
        rows.append(
            {
                "episode_id": eid,
                "path": rel,
                "length": n,
                "metadata": {
                    "sidecar": f"play/{eid}.npz",
                    "sidecar_sha256": _sha(side),
                    "facts": {"attempts": attempts},
                },
            }
        )
        table.append(
            {
                "episode_id": eid,
                "shard": "shard-00",
                "seed": 1000 + k,
                "offset": offset,
                "frames": n,
            }
        )
        actions_all.append(action)
        phase_all.append(phase)
        lat_all.append(rng.normal(size=(n, 32, 384)).astype(np.float16))
        offset += n
    (shard / "meta/jepa_splits.json").write_text(json.dumps({"episodes": eids}))
    sha["meta/jepa_splits.json"] = _sha(shard / "meta/jepa_splits.json")
    (shard / "meta/jepa_manifest.json").write_text(json.dumps({"episodes": rows, "sha256": sha}))
    for split, sl in (("train", slice(0, 2)), ("val", slice(2, 3))):
        tab = table[sl]
        base = tab[0]["offset"]
        tab = [r | {"offset": r["offset"] - base} for r in tab]
        (features / f"episodes_{split}.json").write_text(json.dumps(tab))
        np.save(features / f"actions_{split}.npy", np.concatenate(actions_all[sl]))
        np.save(features / f"phase_{split}.npy", np.concatenate(phase_all[sl]))
        np.save(features / f"latents_{split}.npy", np.concatenate(lat_all[sl]))
    return {"corpus": corpus, "features": features, "lat": lat_all, "tmp": tmp_path}


def test_build_split_windows_inputs_and_latents(synthetic):
    from embodied_jepa import so101_hybrid_runtime as hr

    out = synthetic["tmp"] / "windows"
    r = hr.build_split("train", synthetic["features"], synthetic["corpus"], out)
    assert r["windows"] == 2 and r["frames"] == 2 * (13 + 7)
    lat = np.load(out / "latents_train.npy")
    idx = np.load(out / "index_train.npy")
    store = np.concatenate(synthetic["lat"][:2]).reshape(-1, 12288)
    assert np.array_equal(lat, store[idx])
    x35 = np.load(out / "x35_train.npy")
    assert np.allclose(x35[:20, 18] * 100, np.arange(20))  # the clock restarts at each window
    with pytest.raises(FileExistsError):
        hr.build_split("train", synthetic["features"], synthetic["corpus"], out)
    v = hr.build_split("val", synthetic["features"], synthetic["corpus"], out)
    train_eps = {w["episode_id"] for w in json.loads((out / "windows_train.json").read_text())}
    val_eps = {w["episode_id"] for w in json.loads((out / "windows_val.json").read_text())}
    assert v["windows"] == 1 and not train_eps & val_eps and "e-4" not in train_eps | val_eps


def test_build_split_alignment_and_hash_guards(synthetic):
    from embodied_jepa import so101_hybrid_runtime as hr

    f = synthetic["features"]
    splits = synthetic["corpus"] / "shard-00/meta/jepa_splits.json"
    original = splits.read_text()
    splits.write_text(json.dumps({"episodes": {"train": ["e-1"], "val": [], "test": ["e-2"]}}))
    with pytest.raises(hy.GuardError, match="G-split"):
        hr.build_split("train", f, synthetic["corpus"], synthetic["tmp"] / "w3")
    splits.write_text(original)
    a = np.load(f / "actions_train.npy")
    a[8, 0] += 0.25  # inside the first window
    np.save(f / "actions_train.npy", a)
    with pytest.raises(hy.GuardError, match="G-align"):
        hr.build_split("train", f, synthetic["corpus"], synthetic["tmp"] / "w1")
    side = synthetic["corpus"] / "shard-00/play/e-3.npz"
    side.write_bytes(side.read_bytes() + b"x")
    with pytest.raises(hy.GuardError, match="sidecar"):
        hr.build_split("val", f, synthetic["corpus"], synthetic["tmp"] / "w2")


# ----- A's configuration rule, the approach count (§7.5) --
def test_choose_approach_config_rule():
    zero = dict.fromkeys(hy.CONFIGS, 0)
    rms = dict.fromkeys(hy.CONFIGS, 0.3)
    out = hy.choose_approach_config(zero | {"H16-last": 20}, rms)
    assert out["chosen"] == "H16-last" and not out["approach_fails"]
    out = hy.choose_approach_config(zero | {"H8-last": 10, "H16-mean": 10}, rms | {"H8-last": 0.2})
    assert out["chosen"] == "H8-last" and out["approach_fails"]
    # full tie: mean over last, then H = 8
    assert hy.choose_approach_config(zero, rms)["chosen"] == "H8-mean"
    assert (
        hy.choose_approach_config(zero | {"H8-last": 3, "H16-last": 3}, rms)["chosen"] == "H8-last"
    )
    with pytest.raises(ContractError):
        hy.choose_approach_config({"H8-last": 1}, rms)


def test_handovers_by_tolerance_count():
    logs = [
        {"handover": {"how": "tolerance"}},
        {"handover": {"how": "timeout"}},
        {"handover": None},
        {"handover": {"how": "tolerance"}},
    ]
    assert hy.handovers_by_tolerance(logs) == 2


# ----- statistics and the bar (§7.6, §11) --
def test_bar_rule_and_its_cap():
    r = hy.bar_rule(40, 1.0, 1.0)
    assert r["bar"] == 64 and r["admissible"] and not r["cap_binds"]
    r = hy.bar_rule(56, 1.0, 1.0)
    assert r["bar"] == math.ceil(round(r["lcb"] * 128, 9)) and r["bar"] == 99
    r = hy.bar_rule(56, 48 / 64, 1.0)  # c = 0.675 binds below LCB 0.77
    assert r["cap_binds"] and r["bar"] == math.ceil(0.9 * 0.75 * 128)
    r = hy.bar_rule(60, 34 / 64, 1.0)  # c = 0.478 < 0.50
    assert not r["admissible"]
    table = hy.bar_table()
    assert [row["bar"] for row in table[:4]] == [64, 64, 81, 99]
    assert table[1]["power_at_hd"] > 0.99


def test_bootstrap_and_power_are_fixed_by_their_salts():
    a = np.r_[np.ones(40), np.zeros(24)].astype(bool)
    b = np.r_[np.ones(20), np.zeros(44)].astype(bool)
    lo, hi = hy.bootstrap_difference(a, b)
    assert (lo, hi) == tuple(cl.bootstrap_difference(a, b, salt=10113))
    assert lo > 0
    p = hy.mcnemar_power(0.6, 0.3, "independent", n=128, trials=2000)
    assert p == hy.mcnemar_power(0.6, 0.3, "independent", n=128, trials=2000)
    assert 0.9 < p <= 1.0


# ----- Stage 0 rows (§7.7) and gated rows (§9), both directions --
def test_every_stage0_row_in_both_directions():
    ok = {
        "gloc_eligible": True,
        "k0_p": 30,
        "a_a": 20,
        "sg_d": 60,
        "a_h": 40,
        "wf_d": 60,
        "h_d": 48,
        "s_fits_without_replication": True,
    }
    assert hy.stage0_row(**ok)["row"] == "S0-DONE"
    assert hy.stage0_row(void_reasons=["x"], **ok)["row"] == "S0-VOID"
    assert hy.stage0_row(**ok | {"gloc_eligible": False})["row"] == "S0-GLOC-WEAK"
    assert hy.stage0_row(**ok | {"k0_p": 15})["row"] == "S0-GLOC-WEAK"
    assert hy.stage0_row(**ok | {"k0_p": 16})["row"] == "S0-DONE"
    assert hy.stage0_row(**ok | {"sg_d": 31})["row"] == "S0-GLOC-WEAK"
    assert hy.stage0_row(**ok | {"a_a": 15})["row"] == "S0-APPROACH"
    assert hy.stage0_row(**ok | {"a_a": 16})["row"] == "S0-DONE"
    assert hy.stage0_row(**ok | {"a_h": 31})["row"] == "S0-APPROACH"
    assert hy.stage0_row(**ok | {"wf_d": 31})["row"] == "S0-APPROACH"
    assert hy.stage0_row(**ok | {"sg_d": 35})["row"] == "S0-CEILING"  # c = 0.49
    assert hy.stage0_row(**ok | {"sg_d": 36})["row"] != "S0-CEILING"
    assert hy.stage0_row(**ok | {"h_d": 34})["row"] == "S0-LOW"
    assert hy.stage0_row(**ok | {"s_fits_without_replication": False})["row"] == "S0-OVER"
    # stages that a row stopped do not run
    early = hy.stage0_row(gloc_eligible=True, k0_p=10)
    assert early["row"] == "S0-GLOC-WEAK" and early["stopped_before"] == "A"
    assert hy.stage0_row(gloc_eligible=True, k0_p=30, a_a=3)["row"] == "S0-APPROACH"
    assert hy.stage0_row(gloc_eligible=True, k0_p=30, a_a=30)["row"] is None


def _outcomes(h, **controls):
    n = 128
    out = {"H": np.r_[np.ones(h), np.zeros(n - h)].astype(bool)}
    for arm in hy.CONTROLS:
        k = controls.get(arm, 0)
        out[arm] = np.r_[np.ones(k), np.zeros(n - k)].astype(bool)
    return out


def test_every_gated_row_in_both_directions():
    assert hy.gated_row(None, 64, void=True)["row"] == "VOID"
    assert hy.gated_row(_outcomes(90, **{"G_loc-only": 10}), 64)["row"] == "HY-PASS"
    unspecific = hy.gated_row(_outcomes(90, **{"N_J+G": 85}), 64)
    assert unspecific["row"] == "HY-UNSPECIFIC" and unspecific["failed_tests"] == ["N_J+G"]
    assert hy.gated_row(_outcomes(40, **{"G_loc-only": 39}), 64)["row"] == "HY-LOW"
    assert hy.gated_row(_outcomes(5, rand=4), 64)["row"] == "HY-FAIL"
    assert hy.gated_row(_outcomes(64), 64)["row"] == "HY-PASS"
    assert hy.gated_row(_outcomes(63), 64)["row"] == "HY-LOW"


def test_the_stage_s_projection_drops_replication_first():
    per = dict.fromkeys(hy.S_GPU_ARMS, 30.0)
    out = hy.project_stage_s(
        lock_seconds_per_episode=per, goals_gpu_h=0.1, cpu_evidence_gpu_h=0.05, used_gpu_h=3.0
    )
    assert math.isclose(out["with_replication_gpu_h"], 8 * 30 * 128 / 3600 + 0.15)
    assert out["fits_with_replication"] and not out["drop_replication"]
    tight = hy.project_stage_s(
        lock_seconds_per_episode=per, goals_gpu_h=0.1, cpu_evidence_gpu_h=0.05, used_gpu_h=3.5
    )
    assert tight["drop_replication"] and tight["fits_without_replication"]
    over = hy.project_stage_s(
        lock_seconds_per_episode=per, goals_gpu_h=0.1, cpu_evidence_gpu_h=0.05, used_gpu_h=6.0
    )
    assert not over["fits_without_replication"]


# ----- G_loc selection (§5.4) --
def _curve(vals, stds=0.1):
    return [
        {"update": u, "val_mse": v, "output_std": [stds] * 6}
        for u, v in zip(range(0, 1000 * len(vals), 1000), vals, strict=True)
    ]


def test_gloc_selection_and_primary_rule():
    c = _curve([0.1, 0.05, 0.03, 0.04, 0.045])
    sel = hy.select_gloc(c)
    assert sel["selected"] == 2000 and not sel["last_two_triggered"]
    assert hy.select_gloc(_curve([0.01, 0.05, 0.04]))["selected"] == 2000  # update 0 excluded
    assert hy.select_gloc(_curve([0.1, 0.05, 0.03]))["last_two_triggered"]
    weak = _curve([0.1, 0.05])
    weak[1]["output_std"] = [0.1] * 5 + [0.01]
    assert hy.select_gloc(weak)["selected"] is None
    p = hy.primary_gloc(
        {
            "G-0": {"selected": 1000, "val_mse": 0.02},
            "G-1": {"selected": 2000, "val_mse": 0.01},
            "G-2": {"selected": None},
        }
    )
    assert p == {"primary": "G-1", "replication": ["G-0", "G-2"], "weak": False}
    assert hy.primary_gloc({"G-0": {"selected": None}})["weak"]


# ----- the ledger (§7.3) --
def _entry(check, passes=True, revision="r1", dirty=False, goals=None, utc="2026-10-20T01:00:00Z"):
    return hy.ledger_entry(
        check=check,
        passes=passes,
        revision=revision,
        dirty=dirty,
        finished_utc=utc,
        goals_sha256=goals,
    )


def test_the_ledger_admits_only_with_every_check_at_the_revision():
    once = [_entry(c) for c in hy.ONCE_CHECKS]
    renc = _entry("renc", goals="g1", utc="2026-10-20T02:00:00Z")
    v = hy.ledger_verdict([*once, renc], revision="r1", goals_sha256="g1")
    assert v["ok"] and v["latest_required_utc"] == "2026-10-20T02:00:00Z"
    assert not hy.ledger_verdict([*once, renc], revision="r2", goals_sha256="g1")["ok"]
    assert not hy.ledger_verdict([*once, renc], revision="r1", goals_sha256="g2")["ok"]
    assert not hy.ledger_verdict(once[1:] + [renc], revision="r1", goals_sha256="g1")["ok"]
    dirty = [_entry(c, dirty=c == "gloc") for c in hy.ONCE_CHECKS]
    assert not hy.ledger_verdict([*dirty, renc], revision="r1", goals_sha256="g1")["ok"]
    failing = [*once, renc, _entry("timing", passes=False, revision="r0")]
    assert not hy.ledger_verdict(failing, revision="r1", goals_sha256="g1")["ok"]
    assert hy.ordered_after_checks("2026-10-20T03:00:00Z", v)
    assert not hy.ordered_after_checks("2026-10-20T01:30:00Z", v)


def test_a_cohort_cannot_be_run_twice_and_the_gated_cohort_is_refused(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_task102 as runner
    finally:
        sys.path.pop(0)
    out = tmp_path / "P-episodes"
    out.mkdir()
    argv = ["episodes", "--cohort", "P", "--goals", "x", "--goals-sha256", "y"]
    with pytest.raises(hy.GuardError, match="G-twice"):
        runner.main([*argv, "--arms", "S+G", "--output", str(out)])
    with pytest.raises(hy.GuardError, match="G-gated"):
        runner.main(
            ["episodes", "--cohort", "S", "--goals", "x", "--goals-sha256", "y", "--arms", "H"]
            + ["--output", str(tmp_path / "S")]
        )


def test_goal_stores_are_refused_on_a_hash_mismatch(tmp_path):
    arrays = {k: np.zeros(1) for k in cl.GOAL_ARRAYS}
    sha = cl.save_goal_store(tmp_path, arrays, {"cohort": "debug", "task": hy.TASK})
    with pytest.raises(cl.GuardError, match="G-goals"):
        cl.load_goal_store(tmp_path, "0" * 64)
    assert cl.load_goal_store(tmp_path, sha)["sha256"] == sha


# ----- imports and byte identity (§7.1) --
def test_core_import_stays_torch_free_and_does_not_load_this_task():
    code = (
        "import sys, embodied_jepa, embodied_jepa.config, embodied_jepa.so101_hybrid;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules;"
        "assert 'embodied_jepa.so101_hybrid_runtime' not in sys.modules;"
        "assert 'embodied_jepa.so101_grasp_policy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    code = "import sys, embodied_jepa; assert 'embodied_jepa.so101_hybrid' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    code = (
        "import sys, embodied_jepa.so101_hybrid_runtime;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


UNCHANGED_SHA256 = {  # §7.1: byte-identical
    "src/embodied_jepa/so101_switch_envelope.py": (
        "6cf863d6283f84e8e19fe9aa83240d8cac14325a3bbc03836e0d24e1752314c5"
    ),
    "src/embodied_jepa/schema_planning.py": (
        "bcfc76b45d4697b90188ce5b61909ec036d7059d41bebd07c76462850910796a"
    ),
    "src/embodied_jepa/so101.py": (
        "7e7baf5d24dff380e1199d333d05b0438af87220222475799a722c2706966930"
    ),
    "src/embodied_jepa/so101_simulation.py": (
        "a5143d9e0979ba042d7c9123742dfc634dc8838276d9ffce243daa1a42837ec8"
    ),
    "src/embodied_jepa/so101_tasks.py": (
        "b6fcb852a89217e55ee7a8e8025c3bbdb46c98358ae13711523b4f7ee8f7246e"
    ),
    "src/embodied_jepa/so101_followers.py": (
        "fe086b8980464898f18f47e14e991d389e6796d8e2b854c797179381c43c5828"
    ),
    "src/embodied_jepa/so101_play_corpus.py": (
        "cb8a6d0f72216f06c3d2a82cf2101a5c0ef7a1f0f732ef40e75cdcc3472463c3"
    ),
    "src/embodied_jepa/config.py": (
        "0a6544a0230ecb7a156da03efbafb0759a494bddb85f82dbd9f6d0427bb69e57"
    ),
}


def test_pinned_files_are_byte_identical():
    from embodied_jepa import so101_wm as sw
    from embodied_jepa import so101_wm_runtime as wr

    assert all(wr.frozen_files_check(ROOT).values())  # TASK-100
    assert all(sw.composed_files_check(ROOT).values())
    task101 = json.loads((ROOT / cl.MANIFEST).read_text())
    assert task101["status"] != "FROZEN"  # TASK-101's manifest stays CANDIDATE
    for rel, pin in (task101["files_sha256"] | task101["composed_files_sha256"]).items():
        assert _sha(ROOT / rel) == pin, rel
    for rel, pin in UNCHANGED_SHA256.items():
        assert _sha(ROOT / rel) == pin, rel


# ----- physics: G-zero, S+G's approach, the sentinel around G_loc --
@pytest.fixture
def sim_ok():
    pytest.importorskip("mujoco")
    if not _fetched():
        pytest.skip("pinned SO-101 assets have not been fetched")


def _goals_for(seeds):
    from embodied_jepa import so101_closed_loop_runtime as rt

    demos = [rt.demonstrate(s, render=False) for s in seeds]
    assert all(d["kept"] for d in demos)
    n = max(d["commands"] for d in demos)
    actions = np.full((len(demos), n, 6), np.nan, np.float32)
    for i, d in enumerate(demos):
        actions[i, : d["commands"]] = d["applied"]
    joints = np.stack([np.stack([c["joints"] for c in d["captured"]]) for d in demos])
    return {
        "arrays": {
            "seeds": np.asarray(seeds),
            "demo_actions": actions,
            "demo_lengths": np.array([d["commands"] for d in demos]),
            "joints": joints,
        },
        "report": {
            "demonstrations": {
                str(d["seed"]): {"success": d["success"], "success_step": d["success_step"]}
                for d in demos
            }
        },
    }


def test_g_zero_the_follower_from_handover_reproduces_the_demonstration(sim_ok):
    from embodied_jepa import so101_hybrid_runtime as hr

    goals = _goals_for([103900, 103901, 103902])
    for slot, seed in enumerate((103900, 103901, 103902)):
        assert hr.g_zero_follower(seed, slot, goals)
    goals["arrays"]["demo_actions"][1, 60, 0] += 0.25  # a diverging demonstration fails
    assert not hr.g_zero_follower(103901, 1, goals)


def test_s_plus_g_approach_equals_the_demonstrations_first_40_commands(sim_ok):
    from embodied_jepa import so101_closed_loop_runtime as rt
    from embodied_jepa import so101_hybrid_runtime as hr

    goals = _goals_for([103903])
    robot = rt.make_robot(False)
    try:
        task = rt.lift_task(robot)
        task.reset(103903)
        aims = hr.follower_aims(robot.sim, robot.home)
        approach = hr.ScriptedApproach(aims["q_pre"] + hy.p_offset(103903, 0.0), robot.scales)
        applied = []
        for _ in range(40):
            robot.observe()
            applied.append(robot.execute(approach.act(robot.sim.targets.copy())).applied_action)
            task.evaluate()
        assert np.array_equal(np.asarray(applied), goals["arrays"]["demo_actions"][0, :40])
        # at r = 0.08 the measured arm ends near q_pre + δ
        task.reset(103903)
        delta = hy.p_offset(103903, 0.08)
        approach = hr.ScriptedApproach(aims["q_pre"] + delta, robot.scales)
        for _ in range(40):
            robot.observe()
            robot.execute(approach.act(robot.sim.targets.copy()))
            task.evaluate()
        q = robot.observe().state.values[0, :6]
        assert hy.arm_rms(q, goals["arrays"]["joints"][0, 0]) == pytest.approx(0.08, abs=0.01)
    finally:
        robot.close()


class _Featuriser:
    def encode(self, frames):
        n = len(frames[cl.CAMERAS[0]])
        return np.zeros((n, 32, 384), np.float16), np.zeros((n, hy.LATENT_DIM), np.float32)


class _Gloc:
    """A G_loc stand-in that records exactly what it receives."""

    def __init__(self):
        self.calls = []

    def act(self, z, q, qvel, targets, t):
        self.calls.append((np.shape(z), np.shape(q), np.shape(qvel), np.shape(targets), t))
        return np.zeros(6, np.float32), np.zeros(6, np.float32)


def test_g_loc_never_receives_task_truth_and_the_clock_starts_at_the_handover(sim_ok):
    from embodied_jepa import so101_closed_loop_runtime as rt
    from embodied_jepa import so101_hybrid_runtime as hr

    robot = rt.make_robot(False)
    try:
        task = rt.lift_task(robot)
        task.reset(103904)
        aims = hr.follower_aims(robot.sim, robot.home)
        sentinel = rt.TruthSentinel(robot.sim)
        gloc = _Gloc()
        ctrl = hr.HybridController(
            featuriser=_Featuriser(),
            phase1=hr.ScriptedApproach(aims["q_pre"], robot.scales),
            phase1_kind="scripted",
            fixed_handover=40,
            gloc=gloc,
        )
        for _ in range(45):
            obs = robot.observe()
            q = obs.state.values[0, :6].astype(float)
            qvel = obs.state.values[0, 6:].astype(float)
            with sentinel.controller():
                action, info = ctrl.act(
                    {c: np.zeros((8, 8, 3), np.uint8) for c in cl.CAMERAS},
                    q,
                    qvel,
                    robot.sim.targets.copy(),
                )
            robot.execute(action)
        assert [c[4] for c in gloc.calls] == [0, 1, 2, 3, 4]
        assert gloc.calls[0][:4] == ((hy.LATENT_DIM,), (6,), (6,), (6,))
        assert ctrl.event["command"] == 40

        class Peeking(_Gloc):
            def act(self, z, q, qvel, targets, t):
                robot.sim.task_truth()

        ctrl = hr.HybridController(featuriser=_Featuriser(), gloc=Peeking())
        obs = robot.observe()
        with pytest.raises(cl.GuardError, match="G-truth"), sentinel.controller():
            ctrl.act(
                {c: np.zeros((8, 8, 3), np.uint8) for c in cl.CAMERAS},
                obs.state.values[0, :6],
                obs.state.values[0, 6:],
                robot.sim.targets.copy(),
            )
    finally:
        robot.close()
