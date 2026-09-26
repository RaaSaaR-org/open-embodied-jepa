"""TASK-064 collector and readability gate: guards, test-split refusal, void reports.

Synthetic/unit checks (plus graphics opt-in simulation checks); never evidence of manipulation
or readability. Learned Apple->Plate is still 0 successes.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import look_corpus as lc
from embodied_jepa import observation_reprobe as orp

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load("_collect_apple_look_t", "scripts/collect_apple_look.py")
reader = load("_read_apple_look_t", "scripts/read_apple_look.py")


# ----- the look is reset-independent ------------------------------------------------------------
def test_look_controller_issues_the_constant_whatever_the_robot():
    streams = []
    for robot in (SimpleNamespace(seed=1), SimpleNamespace(seed=2, apple=(0.3, -0.2))):
        controller = collector.LookController()
        rows = []
        while not controller.done:
            base, requested, kind = controller.command(robot)
            assert kind == "none" and np.array_equal(base, requested)
            rows.append(requested)
            controller.advance(None)
        streams.append(np.stack(rows))
    assert np.array_equal(streams[0], streams[1])
    assert np.array_equal(streams[0], orp.look_sequence())
    assert orp.sequence_sha256(streams[0]) == lc.LOOK_SEQUENCE_SHA256
    assert collector.LookController.phase_index == lc.LOOK_PHASE_INDEX


def test_recorder_labels_carry_no_crop_window():
    rec = collector.Recorder()
    rec.rows["window"] = []
    for key in ("apple", "apple_velocity", "plate", "palm"):
        rec.rows[key] = [np.zeros(3)] * 2
    rec.rows["contact"], rec.rows["dropped"] = [False] * 2, [False] * 2
    rec.rows["stages"] = [[False] * 5] * 2
    for key in ("phase", "perturb"):
        rec.steps[key] = [0]
    for key in ("base", "requested", "projected"):
        rec.steps[key] = [np.zeros(14, np.float32)]
    labels = rec.labels()
    assert not any(k.startswith("robot__") for k in labels)
    assert "privileged__apple_position_world" in labels


# ----- the test split is never decoded ---------------------------------------------------------
class FakeStore:
    def __init__(self):
        self.root = Path("/nonexistent")
        self.read = []
        self.manifest = {"episodes": [{"episode_id": "look-1", "path": "x"}]}

    def read_episode(self, episode_id):
        self.read.append(episode_id)
        return episode_id


def test_train_val_reader_refuses_any_other_episode():
    store = FakeStore()
    qa = collector.TrainValReader(store, {"look-1"})
    assert qa.episode("look-1") == "look-1"
    with pytest.raises(lc.GuardError, match="not a train or val"):
        qa.episode("look-47007")
    with pytest.raises(lc.GuardError, match="not train/val"):
        qa.labels({"episode_id": "look-47007", "metadata": {}}, ("robot",))
    assert store.read == ["look-1"]


def test_readability_frames_reader_refuses_any_other_episode():
    frames = reader.CorpusFrames(FakeStore(), {"look-1"})
    with pytest.raises(lc.GuardError, match="Q-split"):
        frames.row("look-47007")
    with pytest.raises(lc.GuardError, match="Q-split"):
        frames.frames("look-47007", (0, 8))


# ----- guards: G-stop, G-hash, G-plan ----------------------------------------------------------
def test_void_reasons():
    ok = {"timed_out": False, "worker_returncodes": [0, 0]}
    assert collector.void_reason_of(ok, [], []) is None
    assert "wall cap" in collector.void_reason_of(ok | {"timed_out": True}, [], [])
    assert "non-zero" in collector.void_reason_of(ok | {"worker_returncodes": [0, -9]}, [], [])
    assert "changed" in collector.void_reason_of(ok, ["src/x.py"], [])
    assert "changed" in collector.void_reason_of(ok, [], ["collector_sha256"])


def test_integrity_counts_runtime_errors_and_inexact_restores():
    plan = {"roots": [{"seed": 1}, {"seed": 2}]}
    workers = [
        {
            "roots": [{"status": "completed"}, {"status": "completed"}],
            "episodes": [
                {"kind": "root", "branch_frame": 50, "restore_exact": True},
                {"kind": "root", "branch_frame": 60, "restore_exact": True},
            ],
        }
    ]
    supervisor = {"timed_out": False, "worker_returncodes": [0]}
    assert collector.integrity_of(plan, workers, supervisor, [], [])["ok"]
    workers[0]["roots"][1]["status"] = "runtime_error: ValueError: x"
    bad = collector.integrity_of(plan, workers, supervisor, [], [])
    assert not bad["ok"] and len(bad["runtime_errors"]) == 1
    workers[0]["roots"][1]["status"] = "completed"
    workers[0]["episodes"][1]["restore_exact"] = False
    assert not collector.integrity_of(plan, workers, supervisor, [], [])["ok"]


def fake_manifest(tmp_path, plan, *, extra=None):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    files = {"scripts/collect_apple_look.py": b"a", "scripts/read_apple_look.py": b"b"}
    files |= extra or {}
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    manifest = {
        "hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "plan": {
            "sha256_rounded_1e-9": lc.plan_sha256_rounded(plan),
            "sha256_macos_arm64": lc.plan_sha256(plan),
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, root, manifest


def test_preflight_passes_and_every_guard_fails(tmp_path):
    plan = lc.make_plan(lc.PILOT_SEEDS[:4])
    path, root, manifest = fake_manifest(tmp_path, plan)
    facts = collector.preflight(plan, "", manifest_path=path, root=root)
    assert facts["hashes_checked"] == 2
    with pytest.raises(lc.GuardError, match="not clean"):
        collector.preflight(plan, " M src/x.py", manifest_path=path, root=root)
    with pytest.raises(lc.GuardError, match="missing"):
        collector.preflight(plan, "", manifest_path=tmp_path / "nope.json", root=root)
    (root / "scripts/read_apple_look.py").write_bytes(b"changed")
    with pytest.raises(lc.GuardError, match="G-hash"):
        collector.preflight(plan, "", manifest_path=path, root=root)
    (root / "scripts/read_apple_look.py").write_bytes(b"b")
    unpinned = dict(
        manifest,
        hashes={
            "scripts/collect_apple_look.py": manifest["hashes"]["scripts/collect_apple_look.py"]
        },
    )
    path.write_text(json.dumps(unpinned))
    with pytest.raises(lc.GuardError, match="not pinned"):
        collector.preflight(plan, "", manifest_path=path, root=root)
    path.write_text(json.dumps(manifest))
    other = lc.make_plan(lc.PILOT_SEEDS[:5])
    with pytest.raises(lc.GuardError, match="G-plan"):
        collector.preflight(other, "", manifest_path=path, root=root)


def test_the_real_manifest_pins_the_frozen_plan_and_both_scripts():
    manifest = json.loads((ROOT / "benchmarks/manifests/apple-look-corpus-v1.json").read_text())
    for script in ("scripts/collect_apple_look.py", "scripts/read_apple_look.py"):
        assert script in manifest["hashes"]
    plan = lc.make_plan(lc.FROZEN_SEEDS)
    assert lc.plan_sha256_rounded(plan) == manifest["plan"]["sha256_rounded_1e-9"]


# ----- void reports -----------------------------------------------------------------------------
def work_dir(tmp_path, supervisor):
    work = tmp_path / "work"
    work.mkdir()
    plan = lc.make_plan(lc.PILOT_SEEDS[:2])
    (work / "plan.json").write_bytes(lc.plan_bytes(plan))
    provenance = {
        "plan_sha256": hashlib.sha256((work / "plan.json").read_bytes()).hexdigest(),
        "plan_sha256_rounded": lc.plan_sha256_rounded(plan),
        "runtime_source_hashes": {},
        "seed_set": "pilot",
    } | collector.tracked_inputs()
    (work / "provenance.json").write_text(json.dumps(provenance))
    (work / "supervisor.json").write_text(json.dumps(supervisor))
    return work


def test_a_guard_stop_writes_a_void_report(tmp_path):
    work = work_dir(tmp_path, {"timed_out": True, "worker_returncodes": [-9]})
    assert collector.finalize(work, tmp_path / "dataset") == 1
    report = json.loads((work / "collection_report.json").read_text())
    assert report["outcome"] == "V" and report["status"] == "void"
    assert "wall cap" in report["void_reason"]
    assert report["learned_apple_to_plate_successes"] == 0
    with pytest.raises(FileExistsError):
        collector.finalize(work, tmp_path / "dataset2")  # never overwrites a report


def test_a_crash_writes_a_void_report_and_reraises(tmp_path):
    work = work_dir(tmp_path, {"timed_out": False, "worker_returncodes": [0]})
    provenance = json.loads((work / "provenance.json").read_text())
    del provenance["runtime_source_hashes"]  # finalize crashes on a missing key
    (work / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(KeyError):
        collector.finalize(work, tmp_path / "dataset")
    report = json.loads((work / "collection_report.json").read_text())
    assert report["outcome"] == "V" and report["void_reason"].startswith("exception: KeyError")
    assert "Traceback" in report["traceback"]


def test_non_finite_values_are_written_as_null_and_listed(tmp_path):
    path = tmp_path / "r.json"
    collector.write_report(path, {"a": float("nan"), "b": [1.0, float("inf")]})
    report = json.loads(path.read_text())
    assert report["a"] is None and report["b"] == [1.0, None]
    assert sorted(report["non_finite_fields"]) == ["/a=nan", "/b/1=inf"]


def test_an_unstarted_root_is_an_early_stop(tmp_path):
    work = work_dir(tmp_path, {"timed_out": False, "worker_returncodes": [0]})
    worker = {"roots": [{"seed": 47900, "status": "not_started_budget"}], "episodes": []}
    (work / "worker-00.json").write_text(json.dumps(worker | {"looks": []}))
    assert collector.finalize(work, tmp_path / "dataset") == 1
    report = json.loads((work / "collection_report.json").read_text())
    assert report["outcome"] == "V" and "never started" in report["void_reason"]


def test_frozen_runs_refuse_smoke_flags(monkeypatch):
    for flags in (["--limit", "3"], ["--allow-dirty"], ["--skip-readability"]):
        argv = ["x", "run", "--seeds", "frozen", "--output", "o", "--work", "w", *flags]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit, match="pilot/smoke only"):
            collector.main()


def test_frozen_runs_refuse_other_workers_and_caps(monkeypatch):
    for flags in (["--workers", "8"], ["--max-seconds", "9000"], ["--worker-seconds", "100"]):
        argv = ["x", "run", "--seeds", "frozen", "--output", "o", "--work", "w", *flags]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit, match="preregistered workers and caps"):
            collector.main()


def test_standalone_finalize_is_refused_for_the_frozen_run(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (work / "provenance.json").write_text(json.dumps({"seed_set": "frozen"}))
    argv = ["x", "finalize", "--work", str(work), "--output", str(tmp_path / "d")]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="not a recovery path"):
        collector.main()


def test_a_crash_during_collection_writes_a_void_report(tmp_path, monkeypatch):
    def boom(*_args):
        raise RuntimeError("Popen failed")

    monkeypatch.setattr(collector, "collect", boom)
    monkeypatch.setattr(collector, "git", lambda *a: "")
    args = SimpleNamespace(
        output=tmp_path / "d",
        work=tmp_path / "w",
        allow_dirty=True,
        seeds="pilot",
        limit=2,
        skip_readability=True,
    )
    with pytest.raises(RuntimeError, match="Popen failed"):
        collector.run(args)
    report = json.loads((tmp_path / "w" / "collection_report.json").read_text())
    assert report["outcome"] == "V" and "Popen failed" in report["void_reason"]


def test_a_guard_inside_the_gate_is_v(tmp_path, monkeypatch):
    work = work_dir(tmp_path, {"timed_out": False, "worker_returncodes": [0]})
    (work / "worker-00.json").write_text(json.dumps({"roots": [], "episodes": [], "looks": []}))
    monkeypatch.setattr(collector, "assemble", lambda *a: SimpleNamespace(manifest_hash="0" * 64))
    monkeypatch.setattr(collector, "acceptance", lambda *a: {})
    monkeypatch.setattr(
        lc, "checks", lambda result: {"checks": {"L1_look_prefix": True}, "all_passed": True}
    )

    def gate(*_a, **_k):
        raise lc.GuardError("Q-render: stored post-look frame differs on 1 roots")

    monkeypatch.setattr(collector, "load_reader", lambda: SimpleNamespace(readability=gate))
    assert collector.finalize(work, tmp_path / "dataset") == 1
    report = json.loads((work / "collection_report.json").read_text())
    assert report["outcome"] == "V" and "Q-render" in report["void_reason"]


# ----- the gate's guards, both directions -------------------------------------------------------
def test_clock_wall_cap_both_ways():
    reader.Clock(0.0, cap=100.0).check("ok")
    with pytest.raises(lc.GuardError, match="global wall cap"):
        reader.Clock(101.0, cap=100.0).check("late")


def test_fold_weight_repro_finite_and_expert_guards():
    reader.check_folds("a" * 64, "a" * 64)
    with pytest.raises(lc.GuardError, match="Q-folds"):
        reader.check_folds("a" * 64, "b" * 64)
    pins = {"encoder": {"pretrained_weights_digest": "p"}, "floors": {"weights_digest": "f"}}
    reader.check_weights({"pretrained_digest": "p", "floor_digest": "f"}, pins)
    for bad in (
        {"pretrained_digest": "x", "floor_digest": "f"},
        {"pretrained_digest": "p", "floor_digest": "x"},
    ):
        with pytest.raises(lc.GuardError, match="G-weights"):
            reader.check_weights(bad, pins)
    first = {"cls": np.ones((2, 3)), "tokens": np.zeros((2, 4))}
    reader.check_repro(first, {k: v.copy() for k, v in first.items()})
    changed = {"cls": np.ones((2, 3)), "tokens": np.full((2, 4), 1e-12)}
    with pytest.raises(lc.GuardError, match="G-repro"):
        reader.check_repro(first, changed)
    assert reader.finite_features(lambda x: x, "P", 5) == 5

    def nonfinite(_x):
        raise reader.ContractError("non-finite cls features")

    with pytest.raises(lc.GuardError, match="G-finite"):
        reader.finite_features(nonfinite, "P", None)

    def other(_x):
        raise reader.ContractError("expected uint8 frames")

    with pytest.raises(reader.ContractError, match="uint8"):
        reader.finite_features(other, "P", None)
    labels = {"collector__base_action": np.arange(9 * 14).reshape(9, 14)}
    assert reader.recorded_first_policy_action(labels, "e")[0] == 8 * 14
    with pytest.raises(lc.GuardError, match="no policy command"):
        reader.recorded_first_policy_action({"collector__base_action": np.zeros((8, 14))}, "e")


# ----- simulation (graphics opt-in) ---------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
def test_the_collector_look_is_identical_from_two_resets_and_leaves_the_apple(tmp_path):
    pytest.importorskip("mujoco")
    from embodied_jepa.task import AppleToPlateTask

    robot = collector.make_robot()
    try:
        runs = []
        for object_xy, plate_xy in (((0.31, -0.21), (0.47, -0.10)), ((0.37, -0.14), (0.50, -0.2))):
            robot.reset(47900, object_xy=object_xy, plate_xy=plate_xy)
            scorer = AppleToPlateTask(robot)
            rec = collector.Recorder()
            rec.observe(robot, None, scorer.evaluate())
            stop, facts = collector.run_look(robot, scorer, rec)
            assert stop is None and facts["applied_max_abs_minus_requested"] == 0.0
            assert facts["apple_xy_move_max_m"] <= lc.LOOK_MOVE_TOLERANCE_M
            assert not facts["hand_contact"] and facts["look_steps_checked"] == lc.LOOK_STEPS
            assert len(rec.rows["onboard"]) == lc.DECISION_FRAME + 1
            assert rec.steps["phase"] == [lc.LOOK_PHASE_INDEX] * lc.LOOK_STEPS
            runs.append((np.asarray(rec.steps["applied"]), facts["post_look_state"]))
        assert np.array_equal(runs[0][0], runs[1][0])
        assert np.array_equal(runs[0][1], runs[1][1])
    finally:
        robot.close()
