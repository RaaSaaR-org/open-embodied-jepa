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
