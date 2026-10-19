"""TASK-101 (SO-101 S5) Stage 0 tests: cohorts, salts, the switching rule, the derangement, the
projector, the cost adapter, N's ties, F against the goal demonstration, the truth sentinel, the
guards, the statistics, the bar rule and every row in both directions. CPU only, tiny synthetic
models; physics tests skip only when MuJoCo or the pinned SO-101 assets are absent."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
SO101_MANIFEST = ROOT / "assets/so101_manifest.json"


def _fetched() -> bool:
    manifest = json.loads(SO101_MANIFEST.read_text())
    root = ROOT / manifest["local_path"]
    return all((root / relative).is_file() for relative in manifest["sha256"])


# ----- cohorts, seeds, salts --
def test_cohorts_are_the_declared_ranges_and_disjoint():
    c = cl.COHORTS
    assert (c["K"].seeds[0], c["K"].seeds[-1], c["K"].reserve) == (101010, 101041, (101042, 101059))
    assert (c["Kp"].seeds[0], c["Kp"].seeds[-1], c["Kp"].reserve) == (
        101060,
        101091,
        (101092, 101099),
    )
    assert (c["D"].seeds[0], c["D"].seeds[-1], c["D"].reserve) == (101100, 101163, (101164, 101199))
    assert (c["S"].seeds[0], c["S"].seeds[-1], c["S"].reserve) == (101200, 101327, (101400, 101463))
    assert (c["Dp"].seeds[0], c["Dp"].seeds[-1], c["Dp"].reserve) == (
        101600,
        101663,
        (101664, 101699),
    )
    assert (c["debug"].seeds[0], c["debug"].seeds[-1]) == (101900, 101999)
    every = [s for x in c.values() for s in (*x.seeds, *x.reserve_seeds)]
    assert len(every) == len(set(every))
    assert set(cl.SALTS.values()) == {10101, 10102, 10103, 10104, 10105, 10106}


def test_seeds_outside_a_cohort_are_refused():
    assert cl.check_seed("D", 101100) == 101100
    assert cl.check_seed("D", 101199) == 101199  # reserve
    for name, seed in (("D", 101200), ("K", 101100), ("S", 101399), ("debug", 101899)):
        with pytest.raises(cl.GuardError, match="G-seed"):
            cl.check_seed(name, seed)


def test_planner_and_rand_seeds_follow_their_salts():
    assert cl.planner_seed(101100) == 10101 * 10**6 + 101100
    a = cl.rand_action(cl.rand_rng(101100))
    b = cl.rand_action(np.random.default_rng([10102, 101100]))
    assert np.array_equal(a, b) and a.dtype == np.float32


def test_the_primary_seed_rule():
    assert cl.primary_seed() == "J-100101" == cl.PRIMARY
    assert cl.primary_seed({"a": 0.2, "b": 0.1}) == "b"


# ----- the derangement --
@pytest.mark.parametrize("n", [2, 3, 32, 64, 128])
def test_the_derangement_has_no_fixed_point_and_is_fixed_by_its_salt(n):
    d = cl.derangement(n)
    assert sorted(d) == list(range(n)) and not np.any(d == np.arange(n))
    assert np.array_equal(d, cl.derangement(n))
    assert n < 8 or not np.array_equal(d, cl.derangement(n, salt=1))


# ----- the switching rule (§5.3) --
def _goal():
    g = np.zeros((4, 6))
    g[:, 5] = [1.7, 1.7, 0.16, 0.16]
    g[1, :5] = 0.5
    g[2, :5] = 0.5
    g[3, :5] = 0.2
    return g


def test_switching_needs_three_consecutive_near_commands():
    g = _goal()
    s = cl.SubgoalSwitcher(g)
    near = g[0].copy()
    near[:5] += 0.039  # RMS 0.039 <= 0.04
    far = g[0].copy()
    far[:5] += 0.041
    assert [s.update(near), s.update(near), s.update(far)] == [0, 0, 0]
    assert [s.update(near), s.update(near)] == [0, 0]
    assert s.update(near) == 1  # the third consecutive near command switches
    assert s.events[0]["how"] == "reached" and s.events[0]["command"] == 5
    # the jaw tolerance binds too
    jaw_off = g[1].copy()
    jaw_off[5] -= 0.151
    assert [s.update(jaw_off) for _ in range(5)] == [1] * 5


def test_switching_times_out_after_60_45_30_and_keeps_the_last_subgoal():
    s = cl.SubgoalSwitcher(_goal())
    far = np.full(6, 9.0)
    ks = [s.update(far) for _ in range(200)]
    assert ks.count(0) == 60 and ks.count(1) == 45 and ks.count(2) == 30 and ks.count(3) == 65
    assert [e["how"] for e in s.events] == ["timeout"] * 3
    assert sum(cl.SWITCH_TIMEOUTS) == 135


def test_switching_moves_one_subgoal_per_command_and_single_goal_starts_at_four():
    g = _goal()
    s = cl.SubgoalSwitcher(g)
    for k in range(3):
        for _ in range(3):
            got = s.update(g[k])
        assert got == k + 1
    single = cl.SubgoalSwitcher(g, single_goal=True)
    assert {single.update(g[0]) for _ in range(70)} == {3} and single.events == []


# ----- the non-learned arms --
def test_pose_action_is_the_followers_law():
    scales = np.full(5, 0.0625, np.float32)
    targets = np.zeros(6)
    goal = np.array([0.01, -1.0, 0.0625, 0.0, 0.03125, 1.7])
    a = cl.pose_action(targets, goal, scales)
    assert np.allclose(a[:5], [0.16, -1.0, 1.0, 0.0, 0.5]) and a[5] == 1.0
    goal[5] = 0.16
    assert cl.pose_action(targets, goal, scales)[5] == -1.0


# ----- statistics, the bar and the rows --
def test_exact_interval_and_mcnemar():
    lo, hi = cl.exact_interval(40, 64)
    assert abs(lo - 0.4952) < 1e-3 and abs(hi - 0.7432) < 1e-3
    assert cl.exact_interval(0, 32)[0] == 0.0 and cl.exact_interval(32, 32)[1] == 1.0
    assert cl.mcnemar_one_sided(0, 0) == 1.0
    assert abs(cl.mcnemar_one_sided(7, 0) - 1 / 128) < 1e-12
    assert cl.mcnemar_one_sided(7, 0) < 0.01 <= cl.mcnemar_one_sided(6, 0)


def test_the_bar_rule_reproduces_the_protocol_table():
    table = {r["w_d"]: r for r in cl.bar_table()}
    assert {w: r["bar"] for w, r in table.items()} == {
        34: 64,
        36: 64,
        40: 64,
        48: 81,
        56: 99,
        60: 109,
        62: 115,
        64: 116,
    }
    assert abs(table[34]["power_at_wd"] - 0.79) < 0.01
    assert abs(table[36]["power_at_wd_minus_0.05"] - 0.65) < 0.01
    assert abs(table[62]["power_at_wd_minus_0.05"] - 0.84) < 0.01
    assert cl.bar_rule(64, r_f=1.0)["b"] == 0.9
    assert cl.bar_rule(10)["b"] == 0.5


def test_bootstrap_difference_is_fixed_by_its_salt():
    rng = np.random.default_rng(0)
    a, b = rng.random(128) < 0.8, rng.random(128) < 0.7
    one, two = cl.bootstrap_difference(a, b), cl.bootstrap_difference(a, b)
    assert one == two and one[0] <= int(a.sum()) - int(b.sum()) <= one[1]
    same = cl.bootstrap_difference(a, a)
    assert same == [0.0, 0.0]


def test_mcnemar_power_is_fixed_and_sensible():
    p = cl.mcnemar_power(0.6, 0.3, "independent", trials=2000)
    assert p == cl.mcnemar_power(0.6, 0.3, "independent", trials=2000)
    assert p > cl.mcnemar_power(0.5, 0.3, "independent", trials=2000)
    assert cl.mcnemar_power(0.5, 0.5, "overlap", trials=500) == 0.0


def test_choose_config_rule():
    c = dict.fromkeys(cl.CONFIGS, 3)
    assert cl.choose_config(c)["chosen"] == "H8-mean"
    c["H16-last"] = 4
    assert cl.choose_config(c)["chosen"] == "H16-last"
    c["H16-mean"] = 4
    assert cl.choose_config(c)["chosen"] == "H16-mean"
    c = dict.fromkeys(cl.CONFIGS, 0)
    out = cl.choose_config(c)
    assert out["null"] and out["chosen"] == "H8-mean"
    c["H16-mean"] = 9
    elig = dict.fromkeys(cl.CONFIGS, True) | {"H16-mean": False}
    assert cl.choose_config(c, elig)["chosen"] == "H8-mean"
    with pytest.raises(ContractError):
        cl.choose_config({"H8-mean": 1})


def _s0(**kw):
    base = {
        "void_reasons": [],
        "d_rejections": 0,
        "k_counts": dict.fromkeys(cl.CONFIGS, 5),
        "w_d": 48,
        "s_fits_without_replication": True,
    }
    return cl.stage0_row(**(base | kw))["row"]


def test_every_stage0_row_in_both_directions():
    assert _s0() == "S0-DONE"
    assert _s0(void_reasons=["latent"]) == "S0-VOID"
    assert _s0(d_rejections=8) == "S0-CEILING" and _s0(d_rejections=7) == "S0-DONE"
    assert _s0(k_counts=dict.fromkeys(cl.CONFIGS, 0)) == "S0-NULL"
    assert _s0(w_d=34) == "S0-LOW" and _s0(w_d=36) == "S0-DONE"
    assert _s0(s_fits_without_replication=False) == "S0-OVER"
    # first match: VOID before CEILING before NULL
    assert _s0(void_reasons=["x"], d_rejections=9) == "S0-VOID"
    assert _s0(d_rejections=9, k_counts=dict.fromkeys(cl.CONFIGS, 0)) == "S0-CEILING"


def _outcomes(k_w, controls):
    w = np.zeros(128, bool)
    w[:k_w] = True
    out = {"W": w}
    for arm in cl.CONTROLS:
        x = np.zeros(128, bool)
        x[: controls.get(arm, 0)] = True
        out[arm] = x
    return out


def test_every_gated_row_in_both_directions():
    assert cl.gated_row(None, 64, void=True)["row"] == "VOID"
    assert cl.gated_row(_outcomes(90, {}), 64)["row"] == "CL-PASS"
    # a control that lifts as often as W -> CL-UNSPECIFIC
    assert cl.gated_row(_outcomes(90, {"W-shuf": 88}), 64)["row"] == "CL-UNSPECIFIC"
    assert cl.gated_row(_outcomes(40, {}), 64)["row"] == "CL-LOW"
    assert cl.gated_row(_outcomes(40, {"hold": 40}), 64)["row"] == "CL-FAIL"
    assert cl.gated_row(_outcomes(5, {}), 64)["row"] == "CL-FAIL"


def test_the_stage_s_projection_drops_replication_first():
    sec = dict.fromkeys(cl.CEM_ARMS, 3600 / 128.0)  # 1 GPU h per arm at speed-up 1
    p = cl.project_stage_s(
        seconds_per_episode=sec, workers_speedup=1.0, goals_gpu_h=0.1, used_gpu_h=14.0
    )
    assert (
        not p["fits_with_replication"] and p["fits_without_replication"] and p["drop_replication"]
    )


# ----- the goal store and guards --
def _store_arrays(n=3):
    rng = np.random.default_rng(0)
    return {
        "seeds": np.arange(101900, 101900 + n),
        "slots": np.arange(n),
        "frames": np.zeros((n, 4, 2, 2, 2, 3), np.uint8),
        "latents_raw": rng.normal(size=(n, 4, 2, 3)).astype(np.float16),
        "joints": rng.normal(size=(n, 4, 6)),
        "cube_cm": np.zeros((n, 4, 3)),
        "site_m": np.zeros((n, 4, 3)),
        "site_xmat": np.zeros((n, 4, 9)),
        "demo_actions": np.zeros((n, 5, 6), np.float32),
        "demo_lengths": np.full(n, 5),
    }


def test_goal_stores_are_refused_on_a_hash_mismatch_and_never_overwritten(tmp_path):
    sha = cl.save_goal_store(tmp_path, _store_arrays(), {"cohort": "debug"})
    assert cl.load_goal_store(tmp_path, sha)["sha256"] == sha
    with pytest.raises(cl.GuardError, match="G-goals"):
        cl.load_goal_store(tmp_path, "0" * 64)
    with pytest.raises(FileExistsError):
        cl.save_goal_store(tmp_path, _store_arrays(), {"cohort": "debug"})
    with (tmp_path / "goals.npz").open("ab") as f:
        f.write(b"x")
    with pytest.raises(cl.GuardError, match="G-goals"):
        cl.load_goal_store(tmp_path)


def test_replacement_takes_the_next_unused_reserve_seed():
    bad = {101010, 101042}

    def demo(seed):
        return seed not in bad, "failed" if seed in bad else "", None

    out = cl.keep_cohort("K", demo)
    assert out["kept"][0] == 101043 and out["kept"][1] == 101011 and len(out["kept"]) == 32
    assert [r["seed"] for r in out["rejected"]] == [101010, 101042] and out["drawn"] == 34


def test_demo_verdict_rejects_short_or_failed_demonstrations():
    caps = [{}] * 4
    assert cl.demo_verdict({"success": True, "commands": 122, "captured": caps}) == (True, "")
    assert not cl.demo_verdict({"success": True, "commands": 119, "captured": caps[:3]})[0]
    assert not cl.demo_verdict({"success": False, "commands": 200, "captured": caps})[0]
    assert not cl.demo_verdict(
        {"success": True, "stopped": True, "commands": 122, "captured": caps}
    )[0]


def test_a_cohort_cannot_be_run_twice_into_the_same_output(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_task101
    finally:
        sys.path.pop(0)
    (tmp_path / "out").mkdir()
    with pytest.raises(cl.GuardError, match="G-twice"):
        run_task101.main(
            [
                "episodes",
                "--cohort",
                "debug",
                "--goals",
                str(tmp_path),
                "--goals-sha256",
                "0" * 64,
                "--arms",
                "W",
                "--output",
                str(tmp_path / "out"),
            ]
        )
    with pytest.raises(cl.GuardError, match="G-gated"):
        run_task101.main(
            [
                "episodes",
                "--cohort",
                "S",
                "--goals",
                str(tmp_path),
                "--goals-sha256",
                "0" * 64,
                "--arms",
                "W",
                "--output",
                str(tmp_path / "s"),
            ]
        )


def test_core_import_stays_torch_free_and_does_not_load_this_task():
    code = (
        "import sys, embodied_jepa, embodied_jepa.config, embodied_jepa.so101_closed_loop;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules;"
        "import importlib; importlib.reload(embodied_jepa);"
        "assert 'embodied_jepa.so101_closed_loop_runtime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
    code = "import sys, embodied_jepa; assert 'embodied_jepa.so101_closed_loop' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_task100_pins_and_this_tasks_unchanged_files_hold():
    from embodied_jepa import so101_wm as sw
    from embodied_jepa import so101_wm_runtime as wr

    assert all(wr.frozen_files_check(ROOT).values())
    assert all(sw.composed_files_check(ROOT).values())
    unchanged = {  # §7.1: byte-identical to origin/main at d628fae
        "src/embodied_jepa/schema_planning.py": "schema_planning",
        "src/embodied_jepa/so101.py": "so101",
        "src/embodied_jepa/so101_simulation.py": "so101_simulation",
        "src/embodied_jepa/so101_tasks.py": "so101_tasks",
        "src/embodied_jepa/so101_followers.py": "so101_followers",
        "src/embodied_jepa/constraints.py": "constraints",
        "src/embodied_jepa/config.py": "config",
    }
    for rel in unchanged:
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == UNCHANGED_SHA256[rel], rel


UNCHANGED_SHA256 = {
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
    "src/embodied_jepa/constraints.py": (
        "b97f82443dedc868bc8da441ce0a737426a8cead673fd9b9aa11cc3680633144"
    ),
    "src/embodied_jepa/config.py": (
        "0a6544a0230ecb7a156da03efbafb0759a494bddb85f82dbd9f6d0427bb69e57"
    ),
}


# ----- physics: projector, F, the sentinel --
@pytest.fixture
def robot():
    pytest.importorskip("mujoco")
    if not _fetched():
        pytest.skip("pinned SO-101 assets have not been fetched")
    from embodied_jepa import so101_closed_loop_runtime as rt

    r = rt.make_robot(False)
    yield r
    r.close()


def test_the_chained_projector_equals_preview_and_execute(robot):
    from embodied_jepa.so101_tasks import SO101CubeLiftTask

    task = SO101CubeLiftTask(robot)
    params = cl.PreviewParams.of(robot)
    rng = np.random.default_rng(3)
    clipped_seen = rate_seen = 0
    for i in range(12):
        task.reset(101900 + i)
        req = rng.uniform(-1, 1, (16, 6)).astype(np.float32)
        if i % 2:
            req = np.sign(req).astype(np.float32)  # saturate: reach the joint limits
        many = np.stack([req, rng.uniform(-1, 1, (16, 6)).astype(np.float32)])[None]
        vec, final = cl.project_sequences(many, robot.sim.targets.copy(), params)
        t = robot.sim.targets.copy()
        chained = []
        for step in req:
            t, applied, reasons = robot.preview(step, t)
            chained.append(applied)
            clipped_seen += any("joint limit" in r for r in reasons)
            rate_seen += any("rate" in r for r in reasons)
        assert np.array_equal(vec[0, 0], np.asarray(chained)) and np.array_equal(final[0, 0], t)
        for k, step in enumerate(req):
            robot.observe()
            result = robot.execute(step)
            if result.status == "stopped":
                break
            assert np.array_equal(result.applied_action, vec[0, 0, k])
    # from targets next to the joint limits, the arm clip binds as well
    for side in (0, 1):
        start = robot.limits[:, side] + (0.01 if side == 0 else -0.01)
        req = np.full((16, 6), -1.0 if side == 0 else 1.0, np.float32)
        vec, _ = cl.project_sequences(req[None], start, params)
        t, chained = start.copy(), []
        for step in req:
            t, applied, reasons = robot.preview(step, t)
            chained.append(applied)
            clipped_seen += any("joint limit" in r for r in reasons)
        assert np.array_equal(vec[0], np.asarray(chained))
    assert rate_seen > 0 and clipped_seen > 0


def test_f_replays_the_goal_demonstration(robot, tmp_path):
    from embodied_jepa import so101_closed_loop_runtime as rt

    demos = [rt.demonstrate(s, render=False) for s in (101900, 101901)]
    assert all(d["kept"] for d in demos)
    n = len(demos)
    length = max(d["commands"] for d in demos)
    actions = np.full((n, length, 6), np.nan, np.float32)
    for i, d in enumerate(demos):
        actions[i, : d["commands"]] = d["applied"]
    arrays = _store_arrays(n) | {
        "seeds": np.array([101900, 101901]),
        "demo_actions": actions,
        "demo_lengths": np.array([d["commands"] for d in demos]),
    }
    report = {
        "cohort": "debug",
        "demonstrations": {
            str(d["seed"]): {"success": d["success"], "success_step": d["success_step"]}
            for d in demos
        },
    }
    sha = cl.save_goal_store(tmp_path, arrays, report)
    goals = cl.load_goal_store(tmp_path, sha)
    out = tmp_path / "F"
    out.mkdir()
    r = rt.run_follower(101901, 1, goals, out / "101901.npz", render=False, started=0.0)
    assert r["equals_demonstration"] and r["success"]
    goals["arrays"]["demo_actions"][1, 5, 0] += 0.5  # a diverging demonstration is a guard failure
    with pytest.raises(cl.GuardError, match="G-F"):
        rt.run_follower(101901, 1, goals, out / "again.npz", render=False, started=0.0)


def test_controllers_never_reach_task_truth(robot):
    from embodied_jepa import so101_closed_loop_runtime as rt

    sentinel = rt.TruthSentinel(robot.sim)
    robot.sim.task_truth()  # the runner may read it
    with pytest.raises(cl.GuardError, match="G-truth"), sentinel.controller():
        robot.sim.task_truth()
    with pytest.raises(cl.GuardError, match="G-truth"), sentinel.controller():
        robot.sim.gripper_contacts()
    with sentinel.controller():
        rt.HoldController(robot.hold_action).act({}, np.zeros(6), robot.sim.targets)
        rt.PoseController(_goal(), robot.scales).act({}, np.zeros(6), robot.sim.targets)
    robot.sim.task_truth()


# ----- the cost adapter and N's ties (tiny model, CPU) --
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    yield
    torch.set_num_threads(original)


def _tiny_dinov2():
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        hidden_size=384,
        num_hidden_layers=1,
        num_attention_heads=6,
        mlp_ratio=1,
        patch_size=14,
        image_size=224,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return Dinov2Model(config).float().eval()


@pytest.fixture
def tiny(monkeypatch):
    from embodied_jepa import so101_wm_runtime as runtime
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
    from embodied_jepa.models import schema_frozen_tokens as sft

    monkeypatch.setattr(
        sft.SchemaFrozenTokenMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2()
    )
    config = {
        "frozen_encoder": "dinov2_small_224",
        "token_count": 4,
        "token_width": 24,
        "latent_dim": 96,
        "camera": "cached_features",
        "max_horizon": 32,
        "hidden_dim": 16,
        "candidate_chunk_size": 5,
    }
    return sft.create(
        "native_jepa",
        state_schema=runtime.state_schema(),
        action_schema=JOINT_DELTA_SO101_V0,
        config=config,
        seed=3,
        metadata={"task": "test"},
    )


def test_the_adapter_matches_predict_features_and_the_mean_wrapper(tiny):
    from embodied_jepa import so101_closed_loop_runtime as rt
    from embodied_jepa import so101_wm as sw

    rng = np.random.default_rng(1)
    z = rng.normal(size=96).astype(np.float32)
    g = rng.normal(size=96).astype(np.float32)
    cand = rng.uniform(-1, 1, (1, 12, 8, 6)).astype(np.float32)
    last = rt.LatentCost(tiny, "last")
    err = last.step_errors(last.predict(z, cand), g)
    pred = tiny.predict_features(np.repeat(z[None], 12, 0), cand[0])
    ref = sw.latent_error(pred.reshape(-1, 96), np.repeat(g[None], 96, 0)).reshape(12, 8)
    assert np.allclose(err, ref, rtol=1e-6, atol=0)
    d_last = last.distance(last.predict(z, cand), g)
    assert d_last.shape == (1, 12, 8) and d_last.dtype == np.float32
    assert np.array_equal(d_last[0], err.astype(np.float32))
    mean = rt.LatentCost(tiny, "mean")
    d_mean = mean.distance(mean.predict(z, cand), g)
    assert np.allclose(d_mean[0, :, -1], err.mean(1), rtol=1e-6)
    assert np.allclose(d_mean[0, :, 2], err[:, :3].mean(1), rtol=1e-6)


def test_n_ties_are_exact_and_the_planner_keeps_its_initial_mean(tiny):
    from embodied_jepa import so101_closed_loop_runtime as rt
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
    from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner

    rng = np.random.default_rng(2)
    z = rng.normal(size=96).astype(np.float32)
    g = rng.normal(size=96).astype(np.float32)
    n = rt.LatentCost(tiny, "mean", action_blind=True)
    cand = rng.uniform(-1, 1, (1, 32, 8, 6)).astype(np.float32)
    d = n.distance(n.predict(z, cand), g)
    assert d.shape == (1, 32, 8) and (d == d[:, :1]).all()
    params = cl.PreviewParams(
        np.array([[-2, 2]] * 5 + [[-0.2, 1.8]], float),
        np.full(5, 0.0625, np.float32),
        -0.17,
        1.75,
        0.25,
    )
    targets = np.array([0, 0, 0, 0, 0, 1.0])
    planner = SchemaCEMPlanner(
        SchemaCEMConfig(
            action_schema=JOINT_DELTA_SO101_V0,
            horizon=8,
            samples=32,
            iterations=3,
            elites=8,
            seed=5,
            project_candidates=True,
        )
    )
    plan = planner.plan(n, z, g, projector=rt.Projector(params, targets))
    assert np.array_equal(plan.requested_actions[0], np.zeros((8, 6), np.float32))
    zero, _ = cl.project_sequences(np.zeros((1, 8, 6), np.float32), targets, params)
    assert np.array_equal(plan.actions[0], zero[0])
    # a synthetic tie: every cost equal -> the initial mean as well
    w = rt.LatentCost(tiny, "last")
    w.step_errors = lambda p, goal: np.ones((32, 8))
    plan = SchemaCEMPlanner(
        SchemaCEMConfig(
            action_schema=JOINT_DELTA_SO101_V0,
            horizon=8,
            samples=32,
            iterations=3,
            elites=8,
            seed=5,
            project_candidates=True,
        )
    ).plan(w, z, g, projector=rt.Projector(params, targets))
    assert np.array_equal(plan.requested_actions[0], np.zeros((8, 6), np.float32))


def test_checkpoints_are_refused_on_a_hash_mismatch(tmp_path, monkeypatch):
    from embodied_jepa import so101_closed_loop_runtime as rt

    (tmp_path / "J-100101").mkdir()
    (tmp_path / "J-100101" / "model.pt").write_bytes(b"not the checkpoint")
    with pytest.raises(cl.GuardError, match="G-checkpoint"):
        rt.load_model("J-100101", "cpu", checkpoints=tmp_path)
    with pytest.raises(ContractError):
        rt.load_model("J-999", "cpu", checkpoints=tmp_path)
