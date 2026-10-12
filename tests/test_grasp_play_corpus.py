"""TASK-090 (docs/experiments/grasp_play_corpus_v1.md): the collector's draws, hold and retry, the
Stage C label and window rules, the combined store, the bars, tests and rows, the seed checks and
the core-import check. NumPy only (no simulator, no torch)."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa import play_corpus as pc
from embodied_jepa import play_corpus_grasp as pg
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]


def truth(on=(True, True, False, False), lift_index=None, contact=False):
    pos = []
    for i, name in enumerate(pc.OBJECT_NAMES):
        if on[i]:
            pos.append([0.35 + 0.05 * i, -0.15, pc.TABLE_TOP_Z + pc.OBJECTS[name]["rest"]])
        else:
            pos.append([*pc.PARKING[name], pc.OBJECTS[name]["rest"]])
    pos = np.array(pos)
    if lift_index is not None:
        pos[lift_index, 2] += 0.05
    gcon = np.zeros(4, bool)
    if contact and lift_index is not None:
        gcon[lift_index] = True
    return {
        "position": pos,
        "grasp_contact": gcon,
        "plate_position": np.array([0.40, -0.20, 0.746]),
    }


class FakeRobot:
    manifest = {"translation_per_step_m": 0.01, "rotation_per_step_rad": 0.05}

    def ee_pose(self, side):
        return np.array([0.3, -0.1, 0.1]), np.eye(3)


# ----- the collector (§2.1) ----------------------------------------------------------------------
def test_modes_and_skills_follow_section_2_1():
    modes, skills = [], []
    for s in range(400):
        p = pg.GraspPlayPolicy(np.random.default_rng([pg.POLICY_SALT, s]))
        modes.append(p.mode)
        p._plan(truth(), None)
        skills.append(p.skill)
    assert set(modes) <= {"scripted", "perturbed"}  # no random mode
    assert abs(np.mean(np.array(modes) == "scripted") - 0.55) < 0.08
    assert "poke" not in skills
    assert set(skills) <= {"pick_place", "push", "wander"}
    assert abs(np.mean(np.array(skills) == "pick_place") - 0.85) < 0.06
    assert pg.HOLD_COMMANDS == (40, 80) and pg.RETRIES == 1
    assert (pc.FAILURE_WEIGHTS, pc.CLOSURE, pc.AIM_SIGMA) == (
        (0.65, 0.10, 0.07, 0.08, 0.05, 0.05),
        (0.85, 1.0),
        0.008,
    )


def _pick_policy(failure="none"):
    for s in range(2000):
        p = pg.GraspPlayPolicy(np.random.default_rng([7, s]))
        p._plan(truth(), None)
        if p.skill == "pick_place" and p.failure == failure:
            return p
    raise AssertionError("no pick drawn")


def test_hold_follows_the_lift_waypoint():
    p = _pick_policy()
    tags = [getattr(w, "tag", "") for w in p.queue]
    k = tags.index("lift")
    lift, hold = p.queue[k], p.queue[k + 1]
    assert np.allclose(hold.target, lift.target) and hold.grasp == lift.grasp > 0.8
    assert 40 <= hold.steps <= 80
    assert len(p.queue) == k + 6  # hold, transfer, place, open, retreat


def test_abort_has_no_lift_and_no_retry():
    p = _pick_policy("abort")
    assert all(getattr(w, "tag", "") != "lift" for w in p.queue)


def test_retry_once_after_a_failed_lift_check():
    p = _pick_policy()
    robot = FakeRobot()
    t = truth()
    p.act(robot, t)  # sets the start heights
    # finish the queue up to the end of the lift waypoint
    while getattr(p.queue[0], "tag", "") != "lift":
        p.queue.pop(0)
    p.queue[0].steps = 1
    p.act(robot, t)
    assert p.pending_check is not None and p.pending_check["attempt"] == 0
    p.act(robot, t)  # the check: no lift state -> the retry replaces the queue
    assert p.checks[-1] == {
        "skill_entry": 0,
        "object": p.checks[-1]["object"],
        "attempt": 0,
        "lifted": False,
    }
    assert p.skill_log[-1] == ["pick_retry", "none"] or tuple(p.skill_log[-1]) == (
        "pick_retry",
        "none",
    )
    assert p.failure == "none" and p.pick["attempt"] == 1
    while getattr(p.queue[0], "tag", "") != "lift":
        p.queue.pop(0)
    p.queue[0].steps = 1
    p.act(robot, t)
    p.act(robot, t)  # the retry's check fails too: no second retry
    assert [c["attempt"] for c in p.checks] == [0, 1]
    assert sum(s[0] == "pick_retry" for s in p.skill_log) == 1


def test_lifted_check_continues_with_the_hold():
    p = _pick_policy()
    robot = FakeRobot()
    p.act(robot, truth())
    while getattr(p.queue[0], "tag", "") != "lift":
        p.queue.pop(0)
    p.queue[0].steps = 1
    p.act(robot, truth())
    index = p.pending_check["index"]
    hold_steps = p.queue[0].steps
    p.act(robot, truth(lift_index=index, contact=True))
    assert p.checks[-1]["lifted"] is True
    assert all(s[0] != "pick_retry" for s in p.skill_log)
    assert p.queue[0].steps == hold_steps - 1


def test_split_rule_uses_salt_9005():
    seeds = list(range(920000, 920100))
    split = pg.split_assignment(seeds)
    assert (len(split["val"]), len(split["test"]), len(split["train"])) == (5, 5, 90)
    ranked = sorted(
        seeds, key=lambda s: __import__("hashlib").sha256(f"9005:{s}".encode()).hexdigest()
    )
    assert split["val"] == sorted(ranked[:5])


def test_collection_seeds():
    assert pg.corpus_seeds("play-v2g") == range(920000, 923200)
    assert pg.corpus_seeds("play-v1p") == range(910000, 913200)
    pg.check_collection_seed("grasp", 930299, debug=True)
    pg.check_collection_seed("v0", 930300, debug=True)
    for bad in (("grasp", 930300, True), ("v0", 920000, False), ("grasp", 910000, False)):
        with pytest.raises(ContractError):
            pg.check_collection_seed(*bad[:2], debug=bad[2])
    assert pg.policy_salt("grasp") == 9003 and pg.policy_salt("v0") == 8701


# ----- Stage C labels (§4) -----------------------------------------------------------------------
def side(frames=20, lift_frame=15, contact=True):
    pos = np.tile(truth()["position"], (frames, 1, 1)).astype(np.float32)
    gcon = np.zeros((frames, 4), bool)
    if lift_frame is not None:
        pos[lift_frame:, 0, 2] += 0.04
        gcon[lift_frame:, 0] = contact
    return {
        "object_position": pos,
        "grasp_contact": gcon,
        "requested": np.zeros((frames - 1, 14), np.float32),
        "palms": np.zeros((frames, 24), np.float32),
    }


def test_lift_windows_follow_task087_windows():
    layout = {"objects": {"apple": [0.35, -0.15], "cube": [0.4, -0.15]}}
    lab = gc.episode_labels(side(lift_frame=15), layout)
    # 20 frames: windows t = 0 .. 11 (TASK-087's window_starts), frames t .. t + 8
    assert lab["windows"] == len(gw.window_starts([20], [0])) == 12
    assert lab["lift_windows"] == 12 - 7  # t = 7 .. 11 reach frame 15
    lab = gc.episode_labels(side(lift_frame=19), layout)
    assert lab["lift_windows"] == 1  # only the last window sees the final frame
    assert gc.dev_convention_counts(side(lift_frame=19))["lift_windows"] == 0
    assert gc.episode_labels(side(contact=False), layout)["lift_windows"] == 0
    # a parked object is never counted
    parked = {"objects": {"cube": [0.4, -0.15]}}
    assert gc.episode_labels(side(lift_frame=15), parked)["lift_windows"] == 0


def test_window_any():
    flags = np.zeros(12, bool)
    flags[5] = True
    assert gc.window_any(flags, 3).tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert gc.window_any(flags[:3], 3).size == 0


def _rows(lifts, windows=100):
    return [{"lift_windows": k, "windows": windows} for k in lifts]


def test_stage_c_bars_and_row():
    assert (gc.C1_BAR, gc.C2_BAR, gc.C3_BAND) == (2.0, 1.5, 0.25)
    assert gc.DECLARED_BARS == {"C1": 2.5, "C2": 1.75}
    v1 = _rows([3] * 100)
    ok = gc.stage_c(v1, _rows([7] * 100), _rows([3] * 100))
    assert ok["row"] == "C-PASS"
    assert ok["checks"]["C1"]["ratio"] == pytest.approx(7 / 3)
    assert ok["checks"]["C2"]["ratio"] == pytest.approx((10 / 200) / (6 / 200))
    low = gc.stage_c(v1, _rows([5] * 100), _rows([3] * 100))
    assert low["row"] == "C-FAIL" and not low["checks"]["C1"]["pass"]
    drift = gc.stage_c(v1, _rows([12] * 100), _rows([2] * 100))
    assert drift["row"] == "C-FAIL" and not drift["checks"]["C3"]["pass"]


def test_debug_ratios_and_projection_shapes():
    v1 = _rows([3, 4, 2, 3] * 50)
    r = gc.debug_ratios(v1, _rows([7, 8] * 30), _rows([3] * 10), resamples=200)
    assert r["C1"]["point"] == pytest.approx(7.5 / 3)
    assert r["C1"]["ci95"][0] <= r["C1"]["point"] <= r["C1"]["ci95"][1]
    p = gc.project_stage_c(v1, _rows([7, 8] * 30), added=100, resamples=100)
    assert 0 <= p["C1"]["share_meeting_bar"] <= 1


# ----- combined stores (§2.2) --------------------------------------------------------------------
def _store(folder, split, frames, seed):
    rng = np.random.default_rng(seed)
    w = gd.SplitWriter(folder, split, sum(frames))
    for i, n in enumerate(frames):
        ep = {
            "episode_id": f"e{seed}-{i}",
            "state": rng.normal(size=(n, gw.STATE_DIM)).astype(np.float32),
            "actions": rng.normal(size=(n, gw.ACTION_DIM)).astype(np.float32),
            "palm": rng.normal(size=(n, 3)).astype(np.float32),
        }
        w.add(ep, rng.normal(size=(n, gw.TOKENS, gw.K)).astype(np.float32), shard="s", seed=i)
    w.close()


def test_combined_store_concatenates_and_gathers(tmp_path):
    _store(tmp_path / "a", "train", [12, 15], 1)
    _store(tmp_path / "b", "train", [11, 20, 13], 2)
    parts = [(tmp_path / "a", "train"), (tmp_path / "b", "train")]
    mem = gc.CombinedStore(parts, mmap=False, chunk_frames=7)
    mm = gc.CombinedStore(parts, mmap=True)
    a = gd.FeatureStore(tmp_path / "a", "train")
    b = gd.FeatureStore(tmp_path / "b", "train")
    assert len(mem.table) == 5 and mem.offsets.tolist() == [0, 12, 27, 38, 58]
    assert np.array_equal(mem.latents[:27], a.latents) and np.array_equal(
        mem.latents[27:], b.latents
    )
    starts = gw.window_starts(mem.frames, mem.offsets)
    assert len(starts) == len(gw.window_starts(a.frames, a.offsets)) + len(
        gw.window_starts(b.frames, b.offsets)
    )
    pick = starts[[0, 3, 5, 9, 15, 30, len(starts) - 1]]
    x, y = mem.gather(pick, gw.WINDOW_FRAMES), mm.gather(pick, gw.WINDOW_FRAMES)
    for k in x:
        assert np.array_equal(x[k], y[k])
    # no window crosses a part or an episode
    owner = np.repeat(np.arange(5), mem.frames)
    assert all(owner[s] == owner[s + gw.TRAIN_HORIZON] for s in starts)


def test_moment_buffer_checks():
    buf = {k: np.arange(3, dtype=np.float32) for k in gc.MOMENT_BUFFERS}
    other = {k: v.copy() for k, v in buf.items()}
    assert gc.same_buffers(buf, other)
    other["palm_scale"][1] = np.nextafter(np.float32(1), np.float32(2))
    assert not gc.same_buffers(buf, other)
    assert gc.buffers_sha256(buf) != gc.buffers_sha256(other)


# ----- statistics, bars and rows (§6, §7.2) ------------------------------------------------------
def test_signflip_matches_enumeration():
    rng = np.random.default_rng(0)
    for _ in range(30):
        d = rng.integers(-3, 4, rng.integers(1, 9))
        nz = d[d != 0]
        total = d.sum()
        sums = [
            sum(s * abs(x) for s, x in zip(signs, nz, strict=True))
            for signs in itertools.product((-1, 1), repeat=len(nz))
        ]
        brute = np.mean(np.array(sums) >= total) if len(nz) else 1.0
        assert gc.signflip_p(d) == pytest.approx(brute)
    assert gc.signflip_p([0, 0, 0]) == 1.0
    assert gc.signflip_p([1] * 7) == pytest.approx(2**-7)


def test_bars():
    assert (gc.pooled_bar(0.40, 64), gc.seed_floor(0.40, 64)) == (78, 13)
    assert (gc.pooled_bar(0.40, 128), gc.seed_floor(0.40, 128)) == (156, 26)


def _arm(rate, n, seeds=3, offset=0):
    out = np.zeros((seeds, n), bool)
    k = int(round(rate * n))
    for i in range(seeds):
        out[i, (np.arange(k) + offset + i) % n] = True
    return out


def test_rows_first_match():
    n = 64
    rand = np.zeros(n, bool)
    g = np.zeros((3, n), bool)
    g[:, :40] = True  # 40 of 64 on every seed, the same resets
    c = np.zeros((3, n), bool)
    c[:, :10] = True
    assert gc.b_row(g, c, rand, 0.40)["row"] == "B-PASS"
    assert gc.b_row(g, c, rand, 0.40, void="x")["row"] == "B-VOID"
    weak = g.copy()
    weak[0, 12:] = False  # one seed below the floor of 13
    assert gc.b_row(weak, c, rand, 0.40)["row"] == "B-COMP"
    assert gc.b_row(g, g.copy(), rand, 0.40)["row"] == "B-SIZE"
    zero = np.zeros((3, n), bool)
    fail = gc.b_row(zero, zero, rand, 0.40)
    assert fail["row"] == "B-FAIL" and gc.clause_fires("B-FAIL")
    small_c = np.zeros((3, n), bool)
    small_c[:, :2] = True
    small_g = np.zeros((3, n), bool)
    small_g[:, :8] = True
    under = gc.b_row(small_g, small_c, rand, 0.40)
    assert under["bootstrap_rate_difference"]["ci"][1] >= gc.DELTA or under["row"] == "B-COMP"
    assert not gc.clause_fires("B-UNDER") and not gc.clause_fires("B-SIZE")
    with pytest.raises(ContractError):
        gc.b_row(g[:2], c[:2], rand, 0.40)


def test_power_rule_and_alternative():
    assert gc.power_rule({64: {1.0: 0.81}, 128: {1.0: 0.99}}) == {"n": 64, "escalate": False}
    assert gc.power_rule({64: {1.0: 0.73}, 128: {1.0: 0.97}}) == {"n": 128, "escalate": False}
    assert gc.power_rule({64: {1.0: 0.5}, 128: {1.0: 0.7}})["escalate"]
    r = gc.power(64, 1.0, sims=200)
    assert r["marginal_rates"][0] == pytest.approx(0.42, abs=0.03)
    assert gc.BS_N == 128 and gc.COHORTS["bs-grasp"] == range(904000, 904128)


def test_e_and_a_prime_rules():
    stat = {str(h): {"copy_ratio": {"ci": [0.5, 0.9]}, "sens": {"ci": [1.1, 1.3]}} for h in (4, 8)}
    assert gc.e_checks(stat, None)["pass"]
    vs_n = {str(h): {"acc_ratio": {"ci": [0.8, 1.01]}} for h in (4, 8)}
    assert not gc.e_checks(stat, vs_n)["pass"]
    assert gc.a2_prime({"close_R1": [True] * 48, "lift_R2": [True] * 47})["meets"] is False
    row = gc.a_prime_row(
        {"v2g": {1: True, 2: False, 3: True}, "v1p": {1: False, 2: False, 3: True}}
    )
    assert row["row"] == "A′-FAIL" and row["failing_arms"] == ["v1p"]


def test_probe_auroc():
    assert gc.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert gc.auroc([0.5, 0.5], [0, 1]) == 0.5
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 5))
    y = (x[:, 0] > 0).astype(float)
    folds = gc.probe_folds(np.arange(60))
    s = gc.probe_scores(x, y, folds)
    assert gc.auroc(s, y) > 0.9


# ----- seeds and cohorts (§10) -------------------------------------------------------------------
def test_seed_checks_and_disjoint_cohorts():
    gc.check_seed("k0-roots", 90031)
    with pytest.raises(ContractError):
        gc.check_seed("k0-roots", 90032)
    gc.check_model_seed("G", 90102)
    gc.check_model_seed("N", 90100)
    for bad in (("N", 90101), ("G", 87100), ("P", 90100)):
        with pytest.raises(ContractError):
            gc.check_model_seed(*bad)
    gc.check_model_seed("G", 90900, debug=True)
    ranges = list(gc.COHORTS.values()) + [
        range(910000, 913200),
        range(920000, 923200),
        range(930000, 930400),
    ]
    for a, b in itertools.combinations(ranges, 2):
        assert not set(a) & set(b)
    assert gc.SALTS == {"shuffle": 9002, "policy": 9003, "probe": 9004, "split": 9005,
                        "bootstrap": 9006}  # fmt: skip
    assert gc.BS_SEEDS["M-v2g"] == (90100, 90101, 90102)
    assert gc.BS_SEEDS["M-v1"] == (87100,)  # 87101 and 87102 dropped by §14's rule (R27.25)


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.grasp_corpus_wm, embodied_jepa.play_corpus_grasp;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
