"""TASK-092 (docs/experiments/grasp_rate_aware_wm_v1.md): the at-limit rule and the equivalence
sets (the clip maps every drawn request back onto the recorded command), the augmentation (p = 0
is bitwise the executed input; p = 0.5 changes only index 13 of at-limit steps; its own generator
leaves TASK-087's sampler untouched), the G-req loss against TASK-087's (torch, pinned LeWM),
Stage R's measures, δ, bars and rows on synthetic data, the A‴ and B rows, the reported rate
measures, the seed ranges and their disjointness, the pins and the core-import check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_rate_aware as ra
from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
LEWM = ROOT / "third_party" / "le-wm"
L = ra.RATE_LIMIT


# ----- the rate clip and the at-limit rule (§4.1) ----------------------------------------------
def test_rate_limit_is_the_manifests():
    manifest = json.loads((ROOT / "configs" / "g1_sim_action.json").read_text())
    assert ra.rate_limit_from_manifest(manifest) == pytest.approx(2 / 11, abs=1e-12)
    assert ra.RATE_LIMIT == 2 / 11


def _closing(n=8, start=-1.0):
    """An executed grasp sequence closing at the limit from ``start`` and then holding at 1."""
    g, out = start, []
    for _ in range(n):
        g = min(1.0, g + L)
        out.append(g)
    return np.array(out)


def test_at_limit_rule():
    e = np.stack([_closing(), -np.ones(8), np.r_[_closing(4), _closing(4, 1.0)]])
    prev = np.array([-1.0, -1.0, np.nan])
    up, down = ra.at_limit(prev, e)
    # closing from -1 reaches 1 after 11 steps: all 8 steps are at the limit upward
    assert up[0].all() and not down[0].any()
    assert not (up[1] | down[1]).any()
    # an unknown predecessor never marks step 1; the hold at 1 is not at the limit
    assert not up[2, 0] and up[2, 1:4].all() and not up[2, 4:].any()
    # a step just outside the tolerance is not at the limit
    up2, _ = ra.at_limit(np.array([0.0]), np.array([[L + 2e-4]]))
    assert not up2.any()
    down3 = ra.at_limit(np.array([0.5]), np.array([[0.5 - L]]))[1]
    assert down3.all()


def test_equivalence_sets_map_back_onto_the_executed_grasp():
    rng = np.random.default_rng(0)
    prev = rng.uniform(-1, 1 - L, 500)
    e_up = prev + L
    lo, hi = ra.equivalent_interval(prev, np.ones(500, bool), np.zeros(500, bool))
    assert np.allclose(lo, prev + L) and np.all(hi == 1.0)
    for u in (0.0, 0.3, 0.999999):
        r = lo + u * (hi - lo)
        assert np.allclose(ra.rate_clip(r, prev), e_up, atol=1e-12)
    prev_d = rng.uniform(-1 + L, 1, 500)
    lo, hi = ra.equivalent_interval(prev_d, np.zeros(500, bool), np.ones(500, bool))
    assert np.all(lo == -1.0) and np.allclose(hi, prev_d - L)
    r = lo + 0.7 * (hi - lo)
    assert np.allclose(ra.rate_clip(r, prev_d), prev_d - L, atol=1e-12)
    # any request beyond the set is clipped to the same executed grasp; one inside it is not
    assert ra.rate_clip(prev[0] + 0.5 * L, prev[0]) != pytest.approx(prev[0] + L)


def _batch(b=16, k=8, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, (b, k, 14)).astype(np.float32)
    prev = np.full(b, -1.0)
    for i in range(b):
        start = -1.0 if i % 3 else 0.0
        g = _closing(k, start) if i % 2 == 0 else np.full(k, start)
        a[i, :, ra.GRASP_DIM] = g.astype(np.float32)
        prev[i] = start
    prev[5] = np.nan  # a window at its episode's first command
    return a, prev


def test_augmentation_p0_is_bitwise_and_p05_changes_only_at_limit_grasps():
    a, prev = _batch()
    aug0 = ra.Augmenter(87900, 0.0)
    out0 = aug0(a, prev)
    assert np.array_equal(out0.view(np.uint32), a.view(np.uint32))
    assert aug0.counts["steps_replaced"] == 0
    aug = ra.Augmenter(87900, 0.5)
    out = aug(a, prev)
    diff = out != a
    assert not np.delete(diff, ra.GRASP_DIM, 2).any()  # only index 13 changes
    grasp = a[:, :, ra.GRASP_DIM].astype(np.float64)
    up, down = ra.at_limit(prev, grasp)
    assert not (diff[:, :, ra.GRASP_DIM] & ~(up | down)).any()  # only at-limit steps
    assert aug.counts["steps_replaced"] == int(diff.sum()) > 0
    assert not diff[5, 0, ra.GRASP_DIM]  # an unknown predecessor: step 1 never replaced
    # selected windows replace every at-limit step; unselected windows none
    replaced_windows = diff[:, :, ra.GRASP_DIM].any(1)
    for i in range(len(a)):
        if replaced_windows[i]:
            assert (diff[i, :, ra.GRASP_DIM] == (up[i] | down[i])).all()
    # every drawn request maps back onto the executed grasp through the clip
    previous = np.concatenate([prev[:, None], grasp[:, :-1]], 1)
    back = ra.rate_clip(out[:, :, ra.GRASP_DIM], previous)
    mask = diff[:, :, ra.GRASP_DIM]
    assert np.abs(back[mask] - grasp[mask]).max() <= ra.AT_LIMIT_TOL


def test_augmenter_stream_does_not_depend_on_p_and_uses_salt_9008():
    a, prev = _batch()
    r1, r2 = ra.Augmenter(87100, 0.0), ra.Augmenter(87100, 0.5)
    for _ in range(3):
        r1(a, prev), r2(a, prev)
    assert r1.rng.random() == r2.rng.random()
    want = np.random.default_rng(np.random.SeedSequence([9008, 87100])).random(16)
    got = ra.Augmenter(87100).rng.random(16)
    assert np.array_equal(want, got)
    assert ra.SALTS["augment"] == 9008 and gw.SALTS["sampler"] == 8714


class _Store:
    def __init__(self):
        self.frames = np.array([30, 25, 40], np.int64)
        self.offsets = np.r_[0, np.cumsum(self.frames)[:-1]]
        n = int(self.frames.sum())
        rng = np.random.default_rng(3)
        self.actions = rng.uniform(-1, 1, (n, 14)).astype(np.float32)


def test_generator_separation_leaves_task087s_windows_unchanged():
    """With p = 0 the training batches and their order equal TASK-087's: the sampler draws the
    same windows whether or not the augmenter runs, and the inputs are the executed actions."""
    store = _Store()
    s1 = gd.WindowSampler(store, 87900)
    s2 = gd.WindowSampler(store, 87900)
    aug = ra.Augmenter(87900, 0.0)
    for _ in range(5):
        w1 = s1.draw()
        w2 = s2.draw()
        prev = ra.window_prev(w2, store.actions, store.offsets)
        acts = store.actions[w2[:, None] + np.arange(8)]
        out = aug(acts, prev)
        assert np.array_equal(w1, w2)
        assert np.array_equal(out, acts)


def test_window_prev():
    store = _Store()
    starts = np.array([0, 1, 30, 31, 55])
    prev = ra.window_prev(starts, store.actions, store.offsets)
    assert np.isnan(prev[0]) and np.isnan(prev[2]) and np.isnan(prev[4])
    assert prev[1] == store.actions[0, 13] and prev[3] == store.actions[30, 13]


# ----- the G-req loss (torch, pinned LeWM) -----------------------------------------------------
@pytest.fixture
def torch_mod():
    torch = pytest.importorskip("torch")
    if not (LEWM / "jepa.py").exists():
        pytest.skip("pinned LeWM source not fetched")
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    os.chdir(ROOT)
    yield torch
    torch.set_num_threads(original)


SMALL = {"depth": 1, "heads": 2, "head_dim": 16, "mlp_dim": 32, "proj_hidden": 32,
         "state_hidden": 16, "inverse_hidden": 16, "readout_hidden": 16}  # fmt: skip


def _models(torch):
    from embodied_jepa import grasp_rate_aware_runtime as rr
    from embodied_jepa import grounded_wm_model as gm

    out = []
    for cls in (gm.GroundedWM, rr.GReqWM):
        torch.manual_seed(0)
        m = cls("G", config=SMALL)
        m.set_moments(np.zeros(gw.STATE_DIM), np.ones(gw.STATE_DIM), np.zeros(3), np.ones(3))
        out.append(m)
    return out


def _tbatch(torch, b=4, t=gw.TRAIN_HORIZON):
    g = torch.Generator().manual_seed(1)
    z = torch.randn(b, t + 1, gw.TOKENS, gw.K, generator=g)
    a = torch.rand(b, t, gw.ACTION_DIM, generator=g) * 2 - 1
    s = torch.randn(b, t + 1, gw.STATE_DIM, generator=g)
    return z, a, s


def test_greq_has_task087s_parameters_and_loss(torch_mod):
    torch = torch_mod
    base, req = _models(torch)
    sb, sr = base.state_dict(), req.state_dict()
    assert sb.keys() == sr.keys() and all(torch.equal(sb[k], sr[k]) for k in sb)
    z, a, s = _tbatch(torch)
    base.train(), req.train()
    lb, mb = base.loss(z, a, s, None)
    lr, mr = req.loss(z, a.clone(), s, None, a.clone())
    assert torch.equal(lb, lr) and mb == mr
    lb.backward(), lr.backward()
    for (k, p), (_, q) in zip(base.named_parameters(), req.named_parameters(), strict=True):
        assert torch.equal(p.grad, q.grad), k


def test_greq_loss_feeds_the_augmented_input_and_the_executed_target(torch_mod):
    torch = torch_mod
    _, req = _models(torch)
    z, a, s = _tbatch(torch)
    a_in = a.clone()
    a_in[:, :, ra.GRASP_DIM] = 1.0
    req.eval()
    with torch.no_grad():
        g = torch.Generator().manual_seed(2)
        for q in req.parameters():  # AdaLN-zero starts action-blind: leave the initialisation
            q.add_(0.05 * torch.randn(q.shape, generator=g))
        l_same, m_same = req.loss(z, a, s, None, a)
        l_in, m_in = req.loss(z, a_in, s, None, a)
        l_both, m_both = req.loss(z, a_in, s, None, a_in)
    # the predictor sees the augmented input: the prediction losses change
    assert m_in["prediction_loss"] != m_same["prediction_loss"]
    # the inverse target is the executed action: only the target differs between these two
    assert m_in["prediction_loss"] == m_both["prediction_loss"]
    assert m_in["inverse_loss"] != m_both["inverse_loss"]


# ----- Stage R (§6.2) on synthetic data --------------------------------------------------------
def _r_store():
    """Two episodes whose grasp closes at the limit from frame 5 and frame 3, and one that holds."""
    frames = np.array([30, 30, 30])
    offsets = np.r_[0, 30, 60]
    actions = np.zeros((90, 14), np.float32)
    actions[:, ra.GRASP_DIM] = -1.0
    for off, at in ((0, 5), (30, 3)):
        g = -1.0
        for t in range(at, 29):
            g = min(1.0, g + L)
            actions[off + t, ra.GRASP_DIM] = g
    return frames, offsets, actions


def test_r_roots_and_execution_identical_requests():
    frames, offsets, actions = _r_store()
    starts = np.array([0, 10, 30, 40, 60, 70])
    mask, prev, executed, up, down = ra.r_roots(starts, offsets, actions)
    # roots at an episode's first command have no predecessor; the holding episode has no limit
    assert mask.tolist() == [False, True, False, True, False, False]
    sat = ra.saturated_requests(executed[mask], up[mask], down[mask])
    rnd = ra.random_requests(executed[mask], prev[mask], up[mask], down[mask])
    assert ra.executed_identically(sat, executed[mask], prev[mask])
    assert ra.executed_identically(rnd, executed[mask], prev[mask])
    assert ra.executed_identically(executed[mask], executed[mask], prev[mask])
    assert (sat[0, :, ra.GRASP_DIM][up[mask][0]] == 1.0).all()
    bad = sat.copy()
    bad[0, 0, 0] += 0.1  # a change outside the grasp channel is not execution-identical
    assert not ra.executed_identically(bad, executed[mask], prev[mask])
    rnd2 = ra.random_requests(executed[mask], prev[mask], up[mask], down[mask])
    assert np.array_equal(rnd, rnd2)  # salt 9009, deterministic


def test_hand_mse_reads_the_hand_joints_only():
    p = np.zeros((2, 28))
    m = np.zeros((2, 28))
    m[:, 0:7] = 5.0
    m[:, 14:] = 5.0
    assert np.all(ra.hand_mse(p, m) == 0)
    m[0, 7:14] = 1.0
    assert ra.hand_mse(p, m).tolist() == [1.0, 0.0]


def _weights(n, eps=10):
    return gw.cluster_bootstrap(np.arange(n) % eps, resamples=400)


def test_delta_and_r_seed_bars():
    rng = np.random.default_rng(0)
    n = 200
    w = _weights(n)
    base = rng.uniform(0.5, 1.5, n)
    per = {87100: base, 87101: base * 1.1, 87102: base * 0.95}
    d = ra.delta(per, w)
    assert len(d["upper_bounds"]) == 6 and d["delta"] >= 0.1
    assert ra.delta({1: base, 2: base}, w)["delta"] == 0.0
    h_exec_c = rng.uniform(0.04, 0.06, n)
    h_sat_c = h_exec_c * 8.0
    good = ra.r_seed(
        h_exec_req=h_exec_c * 1.05, h_sat_req=h_exec_c * 1.2, h_exec_ctrl=h_exec_c,
        h_sat_ctrl=h_sat_c, l_exec_req=base * 1.001, l_exec_ctrl=base, w_r=w, w_all=w,
        delta_l=0.013, delta_h=0.5,
    )  # fmt: skip
    assert good["rho_ctrl"]["value"] == pytest.approx(8.0)
    assert good["r1_bar"] == pytest.approx(1 + 0.25 * 7.0)
    assert good["r1"] and good["r2"] and good["r_pass"] and good["premise"]
    degenerate = ra.r_seed(
        h_exec_req=h_exec_c * 7.0, h_sat_req=h_exec_c * 7.0, h_exec_ctrl=h_exec_c,
        h_sat_ctrl=h_sat_c, l_exec_req=base, l_exec_ctrl=base, w_r=w, w_all=w,
        delta_l=0.013, delta_h=0.5,
    )  # fmt: skip
    assert degenerate["r1"] and not degenerate["r2"]  # R1 met degenerately, R2 excludes it
    latent_regress = ra.r_seed(
        h_exec_req=h_exec_c, h_sat_req=h_exec_c, h_exec_ctrl=h_exec_c, h_sat_ctrl=h_sat_c,
        l_exec_req=base * 1.05, l_exec_ctrl=base, w_r=w, w_all=w, delta_l=0.013, delta_h=0.5,
    )  # fmt: skip
    assert not latent_regress["r2"]
    no_premise = ra.r_seed(
        h_exec_req=h_exec_c, h_sat_req=h_exec_c, h_exec_ctrl=h_exec_c, h_sat_ctrl=h_exec_c * 1.5,
        l_exec_req=base, l_exec_ctrl=base, w_r=w, w_all=w, delta_l=0.013, delta_h=0.5,
    )  # fmt: skip
    assert not no_premise["premise"]


def _seed_row(premise=True, e=True, r=True):
    return {"premise": premise, "e": {"pass": e}, "r_pass": r}


def test_r_rows_first_match():
    ok = {s: _seed_row() for s in ra.MODEL_SEEDS}
    assert ra.r_row(ok)["row"] == "R-PASS"
    assert ra.r_row(ok, void="hash")["row"] == "R-VOID"
    assert ra.r_row(ok | {87101: _seed_row(premise=False, e=False)})["row"] == "R-PREMISE"
    assert ra.r_row(ok | {87101: _seed_row(e=False, r=False)})["row"] == "E-FAIL"
    one = ok | {87101: _seed_row(r=False)}
    assert ra.r_row(one)["row"] == "R-PASS"  # one failing seed stays in M-req's pool
    two = one | {87102: _seed_row(r=False)}
    assert ra.r_row(two)["row"] == "R-FAIL"


def test_a3_rows_count_r_and_a3_failures_together():
    r = {87100: True, 87101: False, 87102: True}
    assert ra.a3_row(r, {87100: True, 87101: True, 87102: True})["row"] == "A‴-PASS"
    row = ra.a3_row(r, {87100: True, 87101: False, 87102: False})
    assert row["row"] == "A‴-FAIL" and row["failing_seeds"] == [87101, 87102]
    assert ra.a3_row(r, {87100: True, 87101: False, 87102: True})["row"] == "A‴-PASS"
    assert ra.a3_row(r, {s: True for s in r}, void="mismatch")["row"] == "A‴-VOID"


def test_executed_plan_grasp_follows_the_clip():
    plan = np.zeros((8, 14))
    plan[:, ra.GRASP_DIM] = 1.0
    g = ra.executed_plan_grasp(plan, -1.0)
    assert g[0] == pytest.approx(-1 + L) and g[7] == pytest.approx(-1 + 8 * L)
    plan[3:, ra.GRASP_DIM] = -1.0
    g = ra.executed_plan_grasp(plan, -1.0)
    assert g[2] == pytest.approx(-1 + 3 * L) and g[3] == pytest.approx(-1 + 2 * L)


# ----- the closed loop (§6.4) ------------------------------------------------------------------
def _arms(n, rates, seed):
    rng = np.random.default_rng(seed)
    return rng.random((3, n)) < np.asarray(rates)[:, None]


def test_b_rows_first_match():
    n = 128
    rnd = np.zeros(n, bool)
    bar = 0.40
    high = _arms(n, [0.6, 0.6, 0.6], 1)
    low = _arms(n, [0.05, 0.05, 0.05], 2)
    assert ra.b_row(high, low, rnd, bar)["row"] == "B-PASS"
    assert ra.b_row(high, low, rnd, bar, void="x")["row"] == "B-VOID"
    mid = _arms(n, [0.2, 0.2, 0.2], 3)
    assert ra.b_row(mid, low, rnd, bar)["row"] == "B-GAIN"
    assert ra.b_row(high, high.copy(), rnd, bar)["row"] == "B-BAR"
    assert ra.b_row(low, low.copy(), rnd, bar)["row"] == "B-FAIL"
    under = _arms(n, [0.12, 0.12, 0.12], 4)
    row = ra.b_row(under, low, rnd, bar)
    assert row["row"] in ("B-GAIN", "B-UNDER")
    assert ra.clause_fires("B-FAIL") and not any(
        ra.clause_fires(r) for r in ("B-VOID", "B-PASS", "B-GAIN", "B-BAR", "B-UNDER")
    )
    assert ra.pooled_bar(0.40, 128) == 156 and ra.seed_floor(0.40, 128) == 26


def test_b_under_when_the_interval_reaches_delta():
    n = 128
    rnd = np.zeros(n, bool)
    g = np.zeros((3, n), bool)
    c = np.zeros((3, n), bool)
    g[:, :15] = True  # M-req on 15 resets, M-v1 on 9 others: not significant, wide interval
    c[:, 15:24] = True
    row = ra.b_row(g, c, rnd, 0.40)
    assert not row["beats_v1"]
    assert row["bootstrap_rate_difference"]["ci"][1] >= ra.DELTA
    assert row["row"] == "B-UNDER" and not ra.clause_fires(row["row"])
    # at the pooled bar but not beating random, nor M-v1: the bracket of §6.4.5 gives B-FAIL
    g = np.zeros((3, n), bool)
    g[:, :60] = True
    rnd = np.zeros(n, bool)
    rnd[:60] = True
    row = ra.b_row(g, g.copy(), rnd, 0.40)
    assert row["at_pooled_bar"] and not row["beats_random"] and not row["beats_v1"]
    assert row["row"] == "B-FAIL" and ra.clause_fires(row["row"])


def test_rate_measures():
    applied = np.zeros((3, 14))
    applied[:, ra.GRASP_DIM] = [-1 + L, -1 + 2 * L, -1 + 2 * L]
    log = [
        {"index": 2, "grasp": [1.0, 1.0]},  # first clipped; both steps off support
        {"index": 2, "grasp": [-1 + 2 * L, -1 + 3 * L]},  # on support
        {"index": 1, "grasp": [-1 + 2 * L, 0.9]},  # second step off support
    ]
    m = ra.rate_measures({"applied": applied.tolist(), "plan_log": log})
    assert m["commands"] == 3 and m["first_grasp_clipped"] == 1
    assert m["grasp_phase_commands"] == 2 and m["grasp_phase_first_grasp_clipped"] == 1
    assert m["plan_steps"] == 6 and m["plan_steps_off_support"] == 2
    assert m["plans_wholly_on_support"] == 1
    s = ra.rate_summary([m], [False])
    assert s["grasp_phase_first_grasp_clipped_share"] == 0.5
    assert s["plan_steps_off_support_share"] == pytest.approx(1 / 3)
    with pytest.raises(ContractError):
        ra.rate_measures({"applied": applied.tolist(), "plan_log": log[:1]})


def test_power_rule():
    rows = [{"n": 128, "sd": 1.0, "power": 0.97}, {"n": 64, "sd": 1.0, "power": 0.7}]
    assert ra.power_rule(rows) == {"n": 128, "power_at_sd1": 0.97, "escalate": False}
    rows[0]["power"] = 0.79
    assert ra.power_rule(rows)["escalate"]
    small = ra.power_table(sims=50)
    assert {(r["n"], r["sd"]) for r in small["rows"]} == {
        (n, sd) for n in (64, 128) for sd in (0.0, 1.0, 2.0)
    }
    assert small["alternative"] == [0.15, 0.05]


# ----- seeds, runs and pins (§9) ---------------------------------------------------------------
def test_seed_ranges_sizes_and_disjointness():
    c = ra.COHORTS
    assert len(c["a3-roots"]) == 64 and c["a3-roots"][0] == 908000
    assert len(c["bd-grasp"]) == 32 and c["bd-grasp"][0] == 90400
    assert len(c["bs-grasp"]) == 128 and c["bs-grasp"][-1] == 909127
    assert set(c["debug-roots"]) | set(c["debug-grasp"]) == set(ra.DEBUG_SEEDS)
    for a, b in __import__("itertools").combinations(c.values(), 2):
        assert not set(a) & set(b)
    for name, r in ra.EARLIER_RANGES.items():
        for mine in c.values():
            assert not set(r) & set(mine), name
    assert ra.part_ranges("bs-grasp", 2) == [range(909000, 909064), range(909064, 909128)]
    assert ra.DETERMINISM == ("M-req", 87100, 4)
    assert ra.SALTS == {"augment": 9008, "random_equivalent": 9009}
    used = set(gc.SALTS.values()) | set(gw.SALTS.values()) | {9007}
    assert not set(ra.SALTS.values()) & used


def test_runs_admitted():
    ra.check_run("bd-grasp", "M-req", 87100)
    ra.check_run("bd-grasp", "M-v1", 87100)
    with pytest.raises(ContractError):
        ra.check_run("bd-grasp", "M-req", 87101)
    for s in ra.MODEL_SEEDS:
        ra.check_run("bs-grasp", "M-req", s)
        ra.check_run("bs-grasp", "M-v1", s)
    ra.check_run("bs-grasp", "ik", None)
    with pytest.raises(ContractError):
        ra.check_run("bs-grasp", "ik", 87100)
    with pytest.raises(ContractError):
        ra.check_run("bs-grasp", "M-req", 87901)
    ra.check_run("debug-grasp", "M-req", 87901, debug_req_seed=87901)
    assert ra.needs_parts("bs-grasp", "M-v1") == 2 and ra.needs_parts("bd-grasp", "M-req") is None
    ra.check_model_seed(87100)
    ra.check_model_seed(87901, debug=True)
    with pytest.raises(ContractError):
        ra.check_model_seed(87901)


def test_frozen_files_match_their_pins():
    assert all(ra.frozen_files_check(ROOT).values())
    assert ra.stage_r_models() == [("G-req", s) for s in ra.MODEL_SEEDS] + [
        ("G-v1", s) for s in ra.MODEL_SEEDS
    ]
    assert ra.CHECKPOINT_SHA256["G-v1-87100"].startswith("17921a7f")
    assert ra.FEATURE_REPORT_SHA256.startswith("f5648a35") and ra.FEATURE_REPORT_SHA256.endswith(
        "7eae"
    )


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.grasp_rate_aware;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_training_driver_refuses_non_frozen_settings():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_task092_t", ROOT / "scripts" /
                                                  "train_task092.py")  # fmt: skip
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class A:
        pairing = False
        debug = False
        seed = 87100
        updates = 60_000
        select_every = 3_000
        p = 0.5
        cap_seconds = 7200.0
        mmap = False
        time_only = False

    assert mod.check_kind(A) == "job"
    for key, value in (("p", 0.0), ("updates", 30_000), ("seed", 87901), ("mmap", True)):
        bad = type("B", (A,), {key: value})
        with pytest.raises((SystemExit, ContractError)):
            mod.check_kind(bad)
    pairing = type("P", (A,), {"pairing": True, "p": 0.0, "updates": 3000})
    assert mod.check_kind(pairing) == "pairing"
    with pytest.raises(SystemExit):
        mod.check_kind(type("P2", (pairing,), {"p": 0.5}))
    debug = type("D", (A,), {"debug": True, "seed": 87900, "p": 0.0, "updates": 300})
    assert mod.check_kind(debug) == "debug"


def test_summarizer_bd_bs_require_clean_prior_rows_and_stage_t_checkpoints(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "summarize_task092_t", ROOT / "scripts" / "summarize_task092.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return str(path)

    t = write("t.json", {"row": "T-DONE", "models": {"G-req-87100": {"checkpoint_sha256": "aa"}}})
    r = write("r.json", {"stage": "r", "row": "R-PASS", "problems": []})
    a = write("a.json", {"stage": "a", "row": "A‴-PASS", "problems": []})
    args = type("A", (), {"t_summary": t, "r_summary": r, "a_summary": a})
    runs = {"M-req-87100": {"arm": "M-req", "model_seed": 87100,
                            "model": {"checkpoint_sha256": "aa"}}}  # fmt: skip
    assert mod.prior_checks(args, runs) == []
    runs["M-req-87100"]["model"]["checkpoint_sha256"] = "bb"
    assert mod.prior_checks(args, runs) == ["M-req-87100: not Stage T's G-req checkpoint"]
    bad_a = write("a2.json", {"stage": "a", "row": "A‴-FAIL", "problems": []})
    args2 = type("B", (args,), {"a_summary": bad_a})
    assert "Stage a did not end A‴-PASS cleanly" in mod.prior_checks(args2, {})
    assert "no Stage T summary" in mod.prior_checks(type("C", (args,), {"t_summary": None}), {})
