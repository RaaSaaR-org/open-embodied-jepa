"""TASK-102's G_loc (torch, CPU): the head's shape, the live path against the batched offline
prediction, checkpoints refused on a hash or implementation mismatch, a short deterministic
training run, and the hybrid episode loop end to end with a tiny G_loc (H-pose+G, S+G, W+F,
G_loc-only and the A approach arm; rendering off)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_jepa import so101_closed_loop as cl  # noqa: E402
from embodied_jepa import so101_grasp_policy as gp  # noqa: E402
from embodied_jepa import so101_hybrid as hy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    yield
    torch.set_num_threads(original)


class _Moments:
    def apply(self, raw):
        return np.asarray(raw, np.float32) * 0.5


def _frames(n=96, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, hy.LATENT_DIM)).astype(np.float16)
    x35 = hy.nonlatent(
        rng.normal(size=(n, 6)), rng.normal(size=(n, 6)), rng.normal(size=(n, 6)), np.arange(n) % 70
    )
    labels = np.clip(rng.normal(size=(n, 6)), -1, 1).astype(np.float32)
    return gp.Frames(raw, x35, labels, _Moments())


def test_head_shape_and_the_live_path_equals_the_batched_prediction(tmp_path):
    head = gp.GLocHead()
    assert sum(p.numel() for p in head.parameters()) == (
        2 * hy.INPUT_DIM + hy.INPUT_DIM * 512 + 512 + 2 * (512 * 512 + 512) + 512 * 6 + 6
    )
    f = _frames()
    moments = gp.InputMoments.fit(f.x35)
    sha = gp.save_checkpoint(tmp_path / "m.pt", head, moments, {"name": "test"})
    policy = gp.load_checkpoint(tmp_path / "m.pt", "cpu", sha)
    z = _Moments().apply(f.raw16[:16].astype(np.float32))
    batched = policy.predict(z, f.x35[:16])
    for i in range(16):
        x = f.x35[i]
        sent, raw = policy.act(z[i], x[:6], x[6:12], x[12:18], int(round(x[18] * 100)))
        assert np.abs(raw - batched[i]).max() <= 1e-4
        assert np.array_equal(sent, np.clip(raw, -1, 1)) and sent.dtype == np.float32
    with pytest.raises(FileExistsError):
        gp.save_checkpoint(tmp_path / "m.pt", head, moments, {})


def test_checkpoints_are_refused_on_a_file_or_implementation_mismatch(tmp_path, monkeypatch):
    head = gp.GLocHead()
    moments = gp.InputMoments(np.zeros(35), np.ones(35))
    sha = gp.save_checkpoint(tmp_path / "m.pt", head, moments, {})
    with pytest.raises(hy.GuardError, match="G-checkpoint"):
        gp.load_checkpoint(tmp_path / "m.pt", "cpu", "0" * 64)
    monkeypatch.setattr(gp, "implementation_sha256", lambda: "f" * 64)
    with pytest.raises(hy.GuardError, match="implementation"):
        gp.load_checkpoint(tmp_path / "m.pt", "cpu", sha)


def test_input_moments_floor_and_the_cosine_schedule():
    m = gp.InputMoments.fit(np.zeros((10, 35)))
    assert np.all(m.std == np.float32(1e-3))
    assert gp.cosine_lr(0) == pytest.approx(3e-4)
    assert gp.cosine_lr(30_000) == pytest.approx(3e-5)
    assert gp.cosine_lr(15_000) == pytest.approx(0.5 * (3e-4 + 3e-5))


def test_a_short_training_run_is_deterministic_and_selects_by_the_rule():
    f, v = _frames(128, 0), _frames(64, 1)
    m = gp.InputMoments.fit(f.x35)
    a = gp.train(0, f, v, m, "cpu", updates=6, eval_every=2)
    b = gp.train(0, f, v, m, "cpu", updates=6, eval_every=2)
    assert [p["val_mse"] for p in a["curve"]] == [p["val_mse"] for p in b["curve"]]
    assert [p["update"] for p in a["curve"]] == [0, 2, 4, 6]
    assert a["selection"] == hy.select_gloc(a["curve"])
    c = gp.train(1, f, v, m, "cpu", updates=2, eval_every=2)
    assert c["init_seed"] != a["init_seed"]


# ----- the hybrid episode loop, end to end, with a tiny G_loc --
@pytest.fixture
def episode_ctx(tmp_path, monkeypatch):
    pytest.importorskip("mujoco")
    manifest = json.loads((ROOT / "assets/so101_manifest.json").read_text())
    if not all((ROOT / manifest["local_path"] / r).is_file() for r in manifest["sha256"]):
        pytest.skip("pinned SO-101 assets have not been fetched")
    from embodied_jepa import so101_closed_loop_runtime as rt
    from embodied_jepa import so101_hybrid_runtime as hr

    original = rt.make_robot
    monkeypatch.setattr(rt, "make_robot", lambda render: original(False))
    seeds = [103910, 103911]
    demos = [rt.demonstrate(s, render=False) for s in seeds]
    n = max(d["commands"] for d in demos)
    actions = np.full((2, n, 6), np.nan, np.float32)
    for i, d in enumerate(demos):
        actions[i, : d["commands"]] = d["applied"]
    goals = {
        "arrays": {
            "seeds": np.asarray(seeds),
            "joints": np.stack([np.stack([c["joints"] for c in d["captured"]]) for d in demos]),
            "demo_actions": actions,
            "demo_lengths": np.array([d["commands"] for d in demos]),
        },
        "report": {
            "demonstrations": {
                str(d["seed"]): {"success": d["success"], "success_step": d["success_step"]}
                for d in demos
            }
        },
    }

    class Featuriser:
        def encode(self, frames):
            k = len(frames[cl.CAMERAS[0]])
            return np.zeros((k, 32, 384), np.float16), np.zeros((k, hy.LATENT_DIM), np.float32)

    head = gp.GLocHead()
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
    policy = gp.GLocPolicy(head, gp.InputMoments(np.zeros(35), np.ones(35)), "cpu")
    ctx = {
        "goals": goals,
        "featuriser": Featuriser(),
        "goal_latents": np.zeros((2, 4, hy.LATENT_DIM), np.float32),
        "gloc": lambda role: policy,
    }
    return ctx, hr, tmp_path


def _spec(arm, out, **kw):
    out.mkdir(exist_ok=True)
    return {"arm": arm, "seed": 103910, "slot": 0, "tau_h": 0.12, "output": str(out)} | kw


def test_the_episode_loop_runs_every_kind_of_arm(episode_ctx):
    ctx, hr, tmp = episode_ctx
    # H-pose+G: the pose follower reaches subgoal 1 and hands over by tolerance
    r = hr.run_episode(_spec("H-pose+G", tmp / "hp"), ctx)
    assert r["handover"]["how"] == "tolerance" and r["handover"]["command"] < 60
    assert r["commands"] == 200 and not r["success"]  # a zero G_loc holds the half-open jaw
    with np.load(tmp / "hp" / "103910.npz") as log:
        assert set(log["phase"][: r["handover"]["command"]]) == {1}
        assert set(log["phase"][r["handover"]["command"] :]) == {2}
        assert log["encoded"].sum() == 200 - r["handover"]["command"]
    # S+G at r = 0: the approach equals the demonstration's first 40 commands (guard G-zero)
    r = hr.run_episode(_spec("S+G", tmp / "sg", level=0.0), ctx)
    assert r["guards"]["g_zero_approach"] and r["handover"]["command"] == 40
    assert r["handover"]["arm_rms_rad"] == pytest.approx(0.0, abs=1e-6)  # float32 state
    r = hr.run_episode(_spec("S+G", tmp / "sg8", level=0.08), ctx)
    assert r["handover"]["arm_rms_rad"] == pytest.approx(0.08, abs=0.01)
    # G_loc-only: phase 2 from the reset, every command
    r = hr.run_episode(_spec("G_loc-only", tmp / "go"), ctx)
    assert r["handover"] is None and r["phase2"]["commands"] == 200
    with pytest.raises(FileExistsError):
        hr.run_episode(_spec("G_loc-only", tmp / "go"), ctx)


def test_w_plus_f_and_the_approach_arm_with_a_pose_phase1(episode_ctx, monkeypatch):
    ctx, hr, tmp = episode_ctx
    # W+F's phase 1 is CEM over J; here its plan is swapped for the pose plan (no model), so the
    # follower-from-hand-over path is exercised from a tolerance hand-over
    monkeypatch.setitem(hy.ARM_PLAN, "W+F", ("pose", "rule", "follower"))
    monkeypatch.setitem(hy.ARM_PLAN, "H-approach", ("pose", "rule", None))
    r = hr.run_episode(_spec("W+F", tmp / "wf"), ctx)
    assert r["guards"]["g_zero_follower"] and r["handover"]["how"] == "tolerance"
    assert r["success"]
    r = hr.run_episode(_spec("H-approach", tmp / "ha"), ctx)
    assert r["termination"] == "handover" and r["commands"] == r["handover"]["command"]
    assert r["closest_rms_to_subgoal1"] <= 0.12
