"""TASK-087 (``grounded_wm*``): tables, statistics, rows, the projection and the arms' wiring.

Synthetic contract evidence only: no corpus, no pretrained weight, no learned property.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
LEWM = ROOT / "third_party" / "le-wm"


# ----- windows, roots and tables ------------------------------------------------------------------
def test_window_starts_stay_inside_episodes():
    frames = np.array([12, 9, 5, 20])
    offsets = np.concatenate([[0], np.cumsum(frames)[:-1]])
    starts = gw.window_starts(frames, offsets)
    for s in starts:
        e = np.searchsorted(offsets, s, side="right") - 1
        assert s + gw.WINDOW_FRAMES <= offsets[e] + frames[e]
    # 12 -> 4 starts, 9 -> 1, 5 -> 0, 20 -> 12
    assert len(starts) == 4 + 1 + 0 + 12


def test_roots_rule():
    frames = np.array([41, 17, 60])  # 40, 16 and 59 commands
    offsets = np.array([0, 41, 58])
    starts, episode = gw.roots(frames, offsets)
    local = starts - offsets[episode]
    assert list(local[episode == 0]) == [0, 10, 20]
    assert list(local[episode == 1]) == [0]
    assert list(local[episode == 2]) == [0, 10, 20, 30, 40]
    assert (local + gw.EVAL_HORIZON <= frames[episode] - 1).all()


def test_tables_use_other_episodes_and_are_fixed():
    episode = np.repeat(np.arange(20), 5)
    wrong = gw.wrong_table(episode)
    table = gw.candidate_table(episode)
    assert (episode[wrong] != episode).all()
    assert (table[:, 0] == np.arange(len(episode))).all()
    assert (episode[table[:, 1:]] != episode[:, None]).all()
    assert all(len(set(r)) == gw.CANDIDATES for r in table)
    assert (gw.candidate_table(episode) == table).all()
    assert (gw.wrong_table(episode) == wrong).all()


def test_true_rank_counts_ties_against():
    scores = np.array([[1.0, 2, 3], [2.0, 1, 3], [1.0, 1, 3]])
    assert list(gw.true_rank(scores)) == [0, 1, 1]


def test_tie_share():
    commands = np.zeros((3, 4, 14))
    commands[1, 2:] = 1.0  # equal to root 0 over the first 2 steps only
    commands[2] = 5.0
    table = np.array([[0, 1], [1, 2], [2, 0]])
    assert gw.tie_share(commands, table, 2) == pytest.approx(1 / 3)
    assert gw.tie_share(commands, table, 3) == 0.0


# ----- statistics ---------------------------------------------------------------------------------
def test_cluster_bootstrap_weights_follow_episodes():
    episode = np.repeat(np.arange(10), 3)
    w = gw.cluster_bootstrap(episode, resamples=50)
    assert w.shape == (50, 30)
    assert (w[:, 0::3] == w[:, 1::3]).all() and (w[:, 0::3] == w[:, 2::3]).all()
    assert (w.sum(1) == 30).all()


def test_ratio_and_difference_stats():
    rng = np.random.default_rng(0)
    episode = np.repeat(np.arange(40), 10)
    w = gw.cluster_bootstrap(episode, resamples=500)
    true = rng.uniform(1, 2, 400)
    wrong = true * 1.5
    s = gw.ratio_stat(wrong, true, w, 0.95)
    assert s["value"] == pytest.approx(1.5) and s["ci"][0] == pytest.approx(1.5)
    d = gw.ratio_difference(wrong, true, true * 1.2, true, w, 0.95)
    assert d["value"] == pytest.approx(0.3) and d["ci"][0] > 0.29
    m = gw.mean_difference(np.ones(400), np.zeros(400), w, 0.95)
    assert m["value"] == 1.0 and m["ci"] == [1.0, 1.0]


def test_per_root_errors_on_a_known_predictor():
    """A predictor that adds the first command's dim 6 to every latent value: with targets made
    by the true commands, the true sequence ranks first and wrong / true > 1."""
    rng = np.random.default_rng(1)
    n, k = 40, gw.CANDIDATES
    episode = np.repeat(np.arange(20), 2)
    start = rng.normal(size=(n, 2, 3)).astype(np.float32)
    commands = rng.uniform(-1, 1, size=(n, gw.EVAL_HORIZON, 14)).astype(np.float32)

    def predict(s0, cmd, _st, hs):
        return {h: s0 + cmd[:, :h, 6].sum(1)[:, None, None] for h in hs}

    targets = predict(start, commands, None, gw.HORIZONS)
    errors, _ = gw.per_root_errors(
        predict,
        start,
        commands,
        targets,
        None,
        gw.wrong_table(episode),
        gw.candidate_table(episode, k),
    )
    for h in gw.HORIZONS:
        assert (errors[h]["true"] == 0).all()
        assert (errors[h]["rank"] == 0).all()
        assert (errors[h]["wrong"] > 0).all()


def test_budget_rule():
    curve = [[u, 1.0 + 1.0 / u] for u in range(2000, 60001, 2000)]
    b = gw.budget_from_probe(curve, 0.03)
    assert b["updates"] % gw.U_STEP == 0 and gw.U_MIN <= b["updates"] <= gw.U_MAX
    assert b["select_every"] * gw.SELECTIONS == b["updates"]
    flat = [[u, 1.0] for u in range(2000, 60001, 2000)]
    assert gw.budget_from_probe(flat, 0.03)["updates"] == gw.U_MIN
    assert gw.budget_from_probe(curve, 1.0)["escalate"]


def test_delta_from_val_is_the_largest_pairwise_upper_bound():
    rng = np.random.default_rng(2)
    episode = np.repeat(np.arange(30), 4)
    w = gw.cluster_bootstrap(episode, resamples=300)
    base = rng.uniform(1, 2, 120)
    val_true = {
        s: {h: base * (1 + 0.01 * i) for h in gw.GATE_HORIZONS}
        for i, s in enumerate(gw.MODEL_SEEDS)
    }
    delta = gw.delta_from_val(val_true, w)
    for h in gw.GATE_HORIZONS:
        assert delta[str(h)] == pytest.approx(0.02, abs=1e-9)
    same = {s: {h: base for h in gw.GATE_HORIZONS} for s in gw.MODEL_SEEDS}
    assert all(v == 0.0 for v in gw.delta_from_val(same, w).values())


def _criteria(all_ok: bool, h1_ok: bool | None = None):
    h1 = all_ok if h1_ok is None else h1_ok
    return {
        str(s): {str(h): {"all": h1 if h == 1 else all_ok} for h in gw.GATE_HORIZONS}
        for s in gw.MODEL_SEEDS
    }


def test_rows_first_match():
    valid = {str(s): {"valid": True, "reasons": []} for s in gw.MODEL_SEEDS}
    fail = _criteria(False)
    assert (
        gw.decide(void_reasons=["x"], plain_checks=valid, criteria={"S": fail, "G": fail})["row"]
        == "P2-VOID"
    )
    invalid = dict(valid) | {"87101": {"valid": False, "reasons": ["P copy ratio at h=1"]}}
    assert (
        gw.decide(
            void_reasons=[], plain_checks=invalid, criteria={"S": _criteria(True), "G": fail}
        )["row"]
        == "P2-PLAIN-INVALID"
    )
    row = gw.decide(void_reasons=[], plain_checks=valid, criteria={"S": fail, "G": _criteria(True)})
    assert row == {"row": "P2-PASS", "arms": ["G"]}
    row = gw.decide(
        void_reasons=[], plain_checks=valid, criteria={"S": _criteria(False, True), "G": fail}
    )
    assert row == {"row": "P2-SHORT", "arms": ["S"]}
    assert (
        gw.decide(void_reasons=[], plain_checks=valid, criteria={"S": fail, "G": fail})["row"]
        == "P2-FAIL"
    )
    # one seed failing one horizon breaks the pass
    almost = _criteria(True)
    almost["87102"]["8"]["all"] = False
    assert (
        gw.decide(void_reasons=[], plain_checks=valid, criteria={"S": almost, "G": fail})["row"]
        == "P2-SHORT"
    )


def test_phase3_model_tie_break():
    row = {"row": "P2-PASS", "arms": ["S", "G"]}
    d = {"S": {str(s): 0.01 for s in gw.MODEL_SEEDS}, "G": {str(s): 0.02 for s in gw.MODEL_SEEDS}}
    assert gw.phase3_model(row, d) == "G"
    assert gw.phase3_model({"row": "P2-FAIL", "arms": []}, d) is None


def test_criteria_met_reads_bounds():
    comp = {
        "1": {
            "d_sens": {"ci": [0.01, 0.1]},
            "d_top1": {"ci": [0.02, 0.2]},
            "acc_ratio": {"ci": [0.9, 1.04]},
            "copy_ratio": {"ci": [0.5, 0.7]},
        }
    }
    assert gw.criteria_met(comp, {"1": 0.05}, 1)["all"]
    assert not gw.criteria_met(comp, {"1": 0.03}, 1)["all"]
    comp["1"]["d_top1"]["ci"][0] = 0.0
    assert not gw.criteria_met(comp, {"1": 0.05}, 1)["b_ranking"]


def test_seed_guards():
    gw.check_model_seed(87100)
    gw.check_model_seed(87900, debug=True)
    with pytest.raises(ContractError):
        gw.check_model_seed(87900)
    with pytest.raises(ContractError):
        gw.check_arm("X")


# ----- projection ---------------------------------------------------------------------------------
def test_projection_matches_a_direct_pca():
    rng = np.random.default_rng(3)
    mix = rng.normal(size=(gw.TOKEN_WIDTH_RAW, gw.TOKEN_WIDTH_RAW)) / 20
    x = rng.normal(size=(300, gw.TOKENS, gw.TOKEN_WIDTH_RAW)) @ mix + rng.normal(
        size=(1, gw.TOKENS, gw.TOKEN_WIDTH_RAW)
    )
    x = x.reshape(300, -1).astype(np.float32)
    acc = gd.ProjectionAccumulator()
    acc.add(x[:120])
    acc.add(x[120:])
    proj = acc.fit()
    z = (x.astype(np.float64) - x.astype(np.float64).mean(0)) / x.astype(np.float64).std(0)
    t = z.reshape(-1, gw.TOKEN_WIDTH_RAW)
    values = np.linalg.eigvalsh(np.cov(t.T, bias=True))[::-1]
    assert proj["eigenvalues"][:5] == pytest.approx(values[:5], rel=1e-6)
    y = gd.project(x, proj)
    assert y.shape == (300, gw.TOKENS, gw.K)
    # the kept components are orthonormal and ordered
    c = proj["components"]
    assert np.allclose(c.T @ c, np.eye(gw.K), atol=1e-8)
    assert proj["explained"] == pytest.approx(values[: gw.K].sum() / values.sum(), rel=1e-6)


def test_projection_hash_round_trip(tmp_path):
    acc = gd.ProjectionAccumulator()
    acc.add(np.random.default_rng(4).normal(size=(50, gw.RAW_DIM)).astype(np.float32))
    proj = acc.fit()
    sha = gd.save_projection(tmp_path / "p.npz", proj)
    back = gd.load_projection(tmp_path / "p.npz", sha)
    assert gd.projection_sha256(back) == sha
    with pytest.raises(FileExistsError):
        gd.save_projection(tmp_path / "p.npz", proj)
    with pytest.raises(ContractError):
        gd.load_projection(tmp_path / "p.npz", "0" * 64)


def test_feature_store_and_sampler(tmp_path):
    frames = [12, 10, 30]
    w = gd.SplitWriter(tmp_path, "train", sum(frames))
    for i, n in enumerate(frames):
        ep = {
            "episode_id": f"play-{i}",
            "state": np.full((n, gw.STATE_DIM), i, np.float32),
            "actions": np.full((n, gw.ACTION_DIM), i, np.float32),
            "palm": np.full((n, 3), i, np.float32),
        }
        w.add(ep, np.full((n, gw.TOKENS, gw.K), i, np.float32), shard="shard-00", seed=i)
    w.close()
    store = gd.FeatureStore(tmp_path, "train")
    s = gd.WindowSampler(store, 87100, batch=200)
    starts = s.draw()
    batch = store.gather(starts, gw.WINDOW_FRAMES)
    # every window holds one episode's frames only
    lat = batch["latents"][:, :, 0, 0]
    assert (lat == lat[:, :1]).all()
    assert (batch["state"][:, :, 0] == lat).all()
    assert batch["actions"].shape == (200, gw.TRAIN_HORIZON, gw.ACTION_DIM)
    again = gd.WindowSampler(store, 87100, batch=200).draw()
    assert (again == starts).all()
    with pytest.raises(FileExistsError):
        gd.SplitWriter(tmp_path, "train", 3)


# ----- the model ----------------------------------------------------------------------------------
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


SMALL = {
    "depth": 1,
    "heads": 2,
    "head_dim": 16,
    "mlp_dim": 32,
    "proj_hidden": 32,
    "state_hidden": 16,
    "inverse_hidden": 16,
    "readout_hidden": 16,
}


def _model(torch, arm):
    from embodied_jepa import grounded_wm_model as gm

    torch.manual_seed(0)
    m = gm.GroundedWM(arm, config=SMALL)
    m.set_moments(np.zeros(gw.STATE_DIM), np.ones(gw.STATE_DIM), np.zeros(3), np.ones(3))
    return m


def _batch(torch, b=4, t=gw.TRAIN_HORIZON):
    g = torch.Generator().manual_seed(1)
    z = torch.randn(b, t + 1, gw.TOKENS, gw.K, generator=g)
    a = torch.rand(b, t, gw.ACTION_DIM, generator=g) * 2 - 1
    s = torch.randn(b, t + 1, gw.STATE_DIM, generator=g)
    palm = torch.randn(b, t + 1, 3, generator=g)
    return z, a, s, palm


@pytest.mark.parametrize("arm", gw.ARMS)
def test_every_arm_trains_one_step(torch_mod, arm):
    torch = torch_mod
    m = _model(torch, arm)
    z, a, s, palm = _batch(torch)
    loss, metrics = m.loss(z, a, s, palm)
    loss.backward()
    assert np.isfinite(metrics["loss"])
    expected = {
        "G": {"joint_change_loss", "inverse_loss"},
        "I": {"inverse_loss"},
        "C": {"readout_loss"},
    }.get(arm, set())
    assert expected <= set(metrics)
    m.eval()
    pred, st = m.rollout(z[:, 0], a, s[:, 0])
    assert pred.shape == (4, gw.TRAIN_HORIZON, gw.TOKENS, gw.K)
    assert (st is not None) == (arm == "G")


def test_rollout_reads_only_the_start_state(torch_mod):
    """S, G and C: the roll-out's signature takes the start state only; changing the later
    measured states of a window cannot change a roll-out."""
    torch = torch_mod
    for arm in ("S", "G", "C"):
        m = _model(torch, arm).eval()
        z, a, s, _ = _batch(torch)
        with torch.no_grad():
            p1, _ = m.rollout(z[:, 0], a, s[:, 0])
            s2 = s.clone()
            s2[:, 1:] += 10.0
            p2, _ = m.rollout(z[:, 0], a, s2[:, 0])
        assert torch.equal(p1, p2)
        with pytest.raises(ContractError):
            m.rollout(z[:, 0], a, None)


def test_state_slot_is_carried_after_the_final_norm(torch_mod):
    """S carries the transformer's state-slot output (after its final LayerNorm, before nothing
    else) as the next state token; pred_proj acts on the 16 visual tokens only."""
    torch = torch_mod
    m = _model(torch, "S").eval()
    z, a, s, _ = _batch(torch)
    with torch.no_grad():
        token = m.state_embed(m.norm_state(s[:, 0]))
        z1, slot = m.step(z[:, 0], a[:, 0], token)
        x = torch.cat((z[:, 0], token[:, None]), 1)
        out = m.predictor(x, m.action_encoder(a[:, :1]))
        assert torch.equal(slot, out[:, gw.TOKENS])
        z2, _ = m.step(z1, a[:, 1], slot)
        pred, _ = m.rollout(z[:, 0], a[:, :2], s[:, 0])
        assert torch.allclose(pred[:, 1], z2, atol=1e-6)


def test_no_action_arm_ignores_commands(torch_mod):
    torch = torch_mod
    m = _model(torch, "N").eval()
    z, a, s, _ = _batch(torch)
    with torch.no_grad():
        p1, _ = m.rollout(z[:, 0], a)
        p2, _ = m.rollout(z[:, 0], -a)
    assert torch.equal(p1, p2)


def test_plain_arm_uses_commands(torch_mod):
    """AdaLN-zero starts with the action path switched off (upstream's initialisation), so the
    weights are perturbed first."""
    torch = torch_mod
    m = _model(torch, "P").eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    z, a, _, _ = _batch(torch)
    with torch.no_grad():
        p1, _ = m.rollout(z[:, 0], a)
        p2, _ = m.rollout(z[:, 0], -a)
    assert not torch.equal(p1, p2)


def test_predict_at_matches_rollout_and_is_chunk_independent(torch_mod):
    torch = torch_mod
    from embodied_jepa import grounded_wm_model as gm

    m = _model(torch, "G").eval()
    rng = np.random.default_rng(5)
    start = rng.normal(size=(9, gw.TOKENS, gw.K)).astype(np.float32)
    cmd = rng.uniform(-1, 1, size=(9, gw.EVAL_HORIZON, 14)).astype(np.float32)
    st = rng.normal(size=(9, gw.STATE_DIM)).astype(np.float32)
    a = gm.predict_at(m, start, cmd, st, (1, 8), chunk=4)
    b = gm.predict_at(m, start, cmd, st, (1, 8), chunk=9)
    assert np.allclose(a[8], b[8], atol=1e-5)
    assert a[1].shape == (9, gw.TOKENS, gw.K)


def test_core_import_stays_light():
    import subprocess
    import sys

    code = (
        "import sys, embodied_jepa, embodied_jepa.grounded_wm, embodied_jepa.grounded_wm_data;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_store_files_are_rehashed(tmp_path):
    w = gd.SplitWriter(tmp_path, "val", 10)
    ep = {
        "episode_id": "play-0",
        "state": np.zeros((10, gw.STATE_DIM), np.float32),
        "actions": np.zeros((10, gw.ACTION_DIM), np.float32),
        "palm": np.zeros((10, 3), np.float32),
    }
    w.add(ep, np.zeros((10, gw.TOKENS, gw.K), np.float32), shard="shard-00", seed=0)
    files = {"val": w.close()}
    assert len(gd.verify_store_files(tmp_path, files, ("val",))) == 5
    np.save(tmp_path / "palm_val.npy", np.ones((10, 3), np.float32))
    with pytest.raises(ContractError):
        gd.verify_store_files(tmp_path, files, ("val",))


def test_corpus_check_refuses_another_corpus_json(tmp_path):
    (tmp_path / "corpus.json").write_text('{"shards": []}')
    with pytest.raises(ContractError):
        gd.verify_corpus(tmp_path)
    assert gd.verify_corpus(tmp_path, pinned=None)["shards"] == {}


def test_iter_episodes_bounds_the_look_ahead(monkeypatch):
    """At most ``ahead`` episodes are submitted beyond the one being consumed, in order."""
    submitted, consumed = [], []

    class Done:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class FakePool:
        def __init__(self, n):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def apply_async(self, fn, args):
            submitted.append(args[2])
            assert len(submitted) - len(consumed) <= 3 + 1  # ahead + the one handed over
            return Done({"episode_id": args[2]})

    class Ctx:
        Pool = FakePool

    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_context", lambda _: Ctx)
    episodes = [("shard-00", f"play-{i}", i) for i in range(10)]
    for ep in gd.iter_episodes("corpus", episodes, workers=2, ahead=3):
        consumed.append(ep["episode_id"])
    assert consumed == [f"play-{i}" for i in range(10)] == submitted
