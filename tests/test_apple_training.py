"""Supervisor tests use synthetic records, never launch apple model training."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_apple_sensor.py"
spec = importlib.util.spec_from_file_location("train_apple_sensor", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, dataset, output = tmp_path / "repo", tmp_path / "data", tmp_path / "out"
    protocol = root / runner.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text("synthetic fixed protocol")
    source = root / "source.py"
    source.write_text("synthetic source")
    manifest = dataset / "meta/jepa_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "splits": {"train": ["train"], "val": ["val"], "test": ["test"]},
                "split_policy": {"group": "session"},
                "action_manifest": {"version": "fixture"},
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "source_identity",
        lambda _: {
            "revision": "a" * 40,
            "dirty": False,
            "python_source_sha256": runner.digest(source),
        },
    )
    return {
        "root": root,
        "dataset": dataset,
        "output": output,
        "expected_source_sha256": runner.digest(source),
        "expected_dataset_sha256": runner.digest(manifest),
        "expected_protocol_sha256": runner.digest(protocol),
    }


def complete_supervisor(fixture):
    def supervise(command, log_path, clock, root, *, on_start=None):
        if "-c" in command:
            Path(command[-1]).write_text(
                json.dumps(
                    {"backend": "sensor_wm", "config": {"image_size": 24, "latent_dim": 1814}}
                )
            )
        else:
            registration = json.loads((fixture["output"] / "registration.json").read_text())
            assert registration["model"]["config"]["latent_dim"] == 1814
            assert registration["command"] == command
            assert registration["model_config_sha256"] == runner.digest(
                fixture["output"] / "model_config.json"
            )
            on_start()
            assert (
                json.loads((fixture["output"] / "supervisor.json").read_text())["started_runs"] == 1
            )
            report = {
                "status": "completed",
                "completed_steps": 3000,
                "best_step": 2700,
                "test_samples_loaded": False,
                "model_config": registration["model"]["config"],
                "budget": runner.FIXED,
                "provenance": {
                    "dataset_hash": fixture["expected_dataset_sha256"],
                    "source_tree_hash": fixture["expected_source_sha256"],
                    "split_hash": registration["split_hash"],
                    "action_hash": registration["action_hash"],
                },
            }
            (fixture["output"] / "sensor.run.json").write_text(json.dumps(report))
            (fixture["output"] / "sensor.pt").write_bytes(b"synthetic best fixture")
            (fixture["output"] / "sensor.latest.pt").write_bytes(b"synthetic latest fixture")
        return {"started": True, "timeout": False, "returncode": 0}

    return supervise


def test_registers_actual_config_before_fixed_training_and_records_artifact_hashes(
    fixture, monkeypatch
):
    monkeypatch.setattr(runner, "supervise", complete_supervisor(fixture))
    report = runner.run(**fixture)
    assert report["status"] == "completed"
    assert (report["planned_runs"], report["started_runs"], report["completed_runs"]) == (1, 1, 1)
    command = report["command"]
    for name in (
        "steps",
        "batch_size",
        "horizon",
        "validation_every",
        "validation_batches",
        "selection",
        "seed",
        "device",
    ):
        assert command[command.index("--" + name.replace("_", "-")) + 1] == str(runner.FIXED[name])
    assert 0 < float(command[command.index("--max-seconds") + 1]) <= 1800
    assert report["artifacts"]["sensor.pt"] == runner.digest(fixture["output"] / "sensor.pt")
    assert not report["test_samples_loaded_by_supervisor"]
    with pytest.raises(FileExistsError):
        runner.run(**fixture)


def test_bad_expected_hash_prevents_any_child_and_preserves_failure(fixture, monkeypatch):
    monkeypatch.setattr(runner, "supervise", lambda *a, **k: pytest.fail("must not launch"))
    report = runner.run(**(fixture | {"expected_dataset_sha256": "0" * 64}))
    assert report["status"] == "failed_integrity_or_orchestration"
    assert report["started_runs"] == report["completed_runs"] == 0
    assert "dataset_sha256" in report["error"]


def test_configuration_imports_share_budget_and_timeout_means_no_training(fixture, monkeypatch):
    monkeypatch.setattr(
        runner, "supervise", lambda *a, **k: {"started": True, "timeout": True, "returncode": -9}
    )
    report = runner.run(**fixture)
    assert report["status"] == "not_started_total_budget"
    assert report["started_runs"] == 0
    assert not (fixture["output"] / "registration.json").exists()


def test_killed_training_retains_partial_report_without_claiming_completion(fixture, monkeypatch):
    normal = complete_supervisor(fixture)

    def supervise(command, log, clock, root, *, on_start=None):
        if "-c" in command:
            return normal(command, log, clock, root)
        on_start()
        (fixture["output"] / "sensor.run.json").write_text(
            json.dumps(
                {"status": "time_budget", "completed_steps": 0, "test_samples_loaded": False}
            )
        )
        return {"started": True, "timeout": True, "returncode": -9}

    monkeypatch.setattr(runner, "supervise", supervise)
    report = runner.run(**fixture)
    assert report["status"] == "supervisor_timeout"
    assert report["started_runs"] == 1 and report["completed_runs"] == 0
    assert report["child_report"]["completed_steps"] == 0
    assert "sensor.run.json" in report["artifacts"]


@pytest.mark.parametrize(
    "mutation",
    ["protocol", "model_config", "child_budget", "missing_checkpoint", "test_loaded", "source"],
)
def test_changes_or_false_child_completion_are_not_accepted(fixture, monkeypatch, mutation):
    normal = complete_supervisor(fixture)

    def supervise(command, log, clock, root, *, on_start=None):
        result = normal(command, log, clock, root, on_start=on_start)
        if "-c" not in command:
            if mutation == "protocol":
                (root / runner.PROTOCOL).write_text("changed")
            elif mutation == "source":
                (root / "source.py").write_text("changed")
            elif mutation == "model_config":
                (fixture["output"] / "model_config.json").write_text("{}")
            elif mutation == "missing_checkpoint":
                (fixture["output"] / "sensor.pt").unlink()
            else:
                path = fixture["output"] / "sensor.run.json"
                value = json.loads(path.read_text())
                if mutation == "child_budget":
                    value["budget"]["steps"] = 5
                else:
                    value["test_samples_loaded"] = True
                path.write_text(json.dumps(value))
        return result

    monkeypatch.setattr(runner, "supervise", supervise)
    report = runner.run(**fixture)
    assert report["status"] == "failed_integrity_or_orchestration"
    assert report["started_runs"] == 1 and report["completed_runs"] == 0


def test_real_subprocess_timeout_and_environment_are_bounded(tmp_path):
    class ShortClock:
        start = time.perf_counter()

        def remaining(self):
            return max(0.0, 0.15 - (time.perf_counter() - self.start))

    result = runner.supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "timeout.log",
        ShortClock(),
        tmp_path,
    )
    assert result["started"] and result["timeout"] and result["returncode"] != 0
    assert time.perf_counter() - ShortClock.start < 3


def test_clock_counts_host_suspension_and_parent_never_imports_torch(monkeypatch):
    wall, monotonic = [0.0], [0.0]
    monkeypatch.setattr(runner.time, "time", lambda: wall[0])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: monotonic[0])
    clock = runner.Clock()
    wall[0] = 1801
    assert clock.remaining() == 0
    assert clock.snapshot()["elapsed_seconds"] == 1801
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys;runpy.run_path(sys.argv[1]);assert 'torch' not in sys.modules",
            str(SCRIPT),
        ],
        check=True,
        timeout=5,
    )


def test_cli_has_no_unregistered_steps_or_budget_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--steps", "1"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("ready", [False, True])
def test_branch_profile_requires_completed_matched_contrast_evidence(fixture, monkeypatch, ready):
    monkeypatch.setattr(runner, "EXPERIMENT", "apple_branch_training_v1")
    (fixture["dataset"] / "branch_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training_ready": True,
                "action_contrast_gate_passed": ready,
                "dataset_sha256": fixture["expected_dataset_sha256"],
            }
        )
    )
    if ready:
        monkeypatch.setattr(runner, "supervise", complete_supervisor(fixture))
    else:
        monkeypatch.setattr(runner, "supervise", lambda *a, **k: pytest.fail("must not launch"))
    report = runner.run(**fixture)
    assert report["status"] == ("completed" if ready else "failed_integrity_or_orchestration")
    if ready:
        assert (
            json.loads((fixture["output"] / "registration.json").read_text())["experiment"]
            == runner.EXPERIMENT
        )
    else:
        assert report["started_runs"] == 0


def prepare_h16_fixture(fixture, monkeypatch):
    experiment, protocol, seconds, fixed = runner.profile_settings("branches_h16_v1")
    monkeypatch.setattr(runner, "EXPERIMENT", experiment)
    monkeypatch.setattr(runner, "PROTOCOL", protocol)
    monkeypatch.setattr(runner, "MAX_SECONDS", seconds)
    monkeypatch.setattr(runner, "FIXED", fixed)
    path = fixture["root"] / protocol
    path.write_text("synthetic H16 prospective protocol")
    fixture = fixture | {"expected_protocol_sha256": runner.digest(path)}
    manifest_path = fixture["dataset"] / "meta/jepa_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episodes"] = []
    for split in ("train", "val"):
        parent = split
        branch = split + "-branch"
        manifest["splits"][split].append(branch)
        manifest["episodes"].extend(
            [
                {"episode_id": parent, "session_id": parent, "length": 33, "metadata": {}},
                {
                    "episode_id": branch,
                    "session_id": parent,
                    "length": 17,
                    "metadata": {"parent_episode_id": parent},
                },
            ]
        )
    manifest["episodes"].append(
        {"episode_id": "test", "session_id": "test", "length": 17, "metadata": {}}
    )
    manifest_path.write_text(json.dumps(manifest))
    fixture["expected_dataset_sha256"] = runner.digest(manifest_path)
    monkeypatch.setattr(runner, "H16_DATASET_SHA", fixture["expected_dataset_sha256"])
    (fixture["dataset"] / "branch_report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training_ready": True,
                "action_contrast_gate_passed": True,
                "dataset_sha256": fixture["expected_dataset_sha256"],
            }
        )
    )
    return fixture


@pytest.mark.parametrize("mutate_groups", [False, True])
def test_h16_profile_freezes_groups_before_child_and_checks_frozen_bytes(
    fixture, monkeypatch, mutate_groups
):
    fixture = prepare_h16_fixture(fixture, monkeypatch)
    complete = complete_supervisor(fixture)

    def supervised(command, log, clock, root, *, on_start=None):
        result = complete(command, log, clock, root, on_start=on_start)
        if "-m" in command:
            registration = json.loads((fixture["output"] / "registration.json").read_text())
            sampling = registration["sampling"]
            path = Path(command[command.index("--sampling-groups") + 1])
            assert sampling["sha256"] == runner.digest(path)
            assert sampling["quotas_per_batch"] == {"train": [8, 8], "val": [8, 8]}
            assert sampling["groups"] == {
                "train": [["train"], ["train-branch"]],
                "val": [["val"], ["val-branch"]],
            }
            assert registration["fixed"]["horizon"] == 16
            childpath = fixture["output"] / "sensor.run.json"
            child = json.loads(childpath.read_text())
            child["provenance"]["sampling_groups_hash"] = sampling["groups_sha256"]
            child["sampling"] = {"groups": sampling["groups"]}
            child["selection"] = {"horizon": 16, "method": "raw_mse"}
            childpath.write_text(json.dumps(child))
            if mutate_groups:
                path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(runner, "supervise", supervised)
    report = runner.run(**fixture)
    assert report["status"] == (
        "failed_integrity_or_orchestration" if mutate_groups else "completed"
    )
    if mutate_groups:
        assert "registered configuration changed" in report["error"]


def test_h16_group_derivation_rejects_cross_split_parent_and_empty_group():
    manifest = {
        "splits": {"train": ["a", "b"], "val": ["c", "d"], "test": ["z"]},
        "episodes": [
            {
                "episode_id": name,
                "session_id": parent or name,
                "length": 17,
                "metadata": {"parent_episode_id": parent} if parent else {},
            }
            for name, parent in [("a", None), ("b", "a"), ("c", None), ("d", "c"), ("z", None)]
        ],
    }
    assert runner.derive_sampling_groups(manifest)["train"] == [["a"], ["b"]]
    manifest["episodes"][1]["metadata"]["parent_episode_id"] = "z"
    with pytest.raises(ValueError, match="same allowed split"):
        runner.derive_sampling_groups(manifest)
    manifest["episodes"][1]["metadata"] = {}
    with pytest.raises(ValueError, match="original and intervention"):
        runner.derive_sampling_groups(manifest)


def test_new_profile_cli_preserves_older_profiles(monkeypatch, tmp_path):
    original = runner.BASE_FIXED.copy()
    assert runner.profile_settings("branches_v1")[3] == original
    assert runner.profile_settings("sensor_v1")[3] == original
    for name in ("EXPERIMENT", "PROTOCOL", "MAX_SECONDS", "FIXED"):
        monkeypatch.setattr(runner, name, getattr(runner, name))

    def fake_run(*args, **kwargs):
        assert runner.FIXED == original | {"horizon": 16}
        assert runner.MAX_SECONDS == 600 and runner.EXPERIMENT == "apple_branch_training_h16_v1"
        return {
            "status": "completed",
            "timing": {},
            "planned_runs": 1,
            "started_runs": 1,
            "completed_runs": 1,
        }

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--profile",
            "branches_h16_v1",
            "--dataset",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
            "--source-sha256",
            "a" * 64,
            "--dataset-sha256",
            "b" * 64,
            "--protocol-sha256",
            "c" * 64,
        ],
    )
    assert runner.main() == 0
