"""Synthetic assembly and process supervision checks; no learned training evidence."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
pytest.importorskip("PIL")

from embodied_jepa.contracts import ContractError  # noqa: E402
from embodied_jepa.data import DatasetStore, deterministic_fixture  # noqa: E402


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assembly = load_script("assemble_mvp_data")
orchestration = load_script("train_mvp")


@pytest.fixture(autouse=True)
def stable_test_source(monkeypatch):
    # Other workers may edit the shared checkout; tests control their own identity.
    monkeypatch.setattr(orchestration, "identity", lambda: {"fixture": "stable"})


def corpus(root, *, prefix="a", reverse=False, apple=False):
    episode = deterministic_fixture()
    store = DatasetStore.create(
        root,
        fps=20,
        state_schema=episode.state_schema,
        action_manifest={"fixture_scale": 0.01},
        provenance={"license": "Apache-2.0", "software_fixture_only": True, "name": prefix},
    )
    for index in range(3):
        store.write_episode(
            replace(
                episode,
                episode_id=f"episode-{index}",
                session_id=f"{prefix}-{index}",
                object_id="apple" if apple else "cube",
                container_id="plate" if apple else "target",
            )
        )
    splits = {"train": ["episode-0"], "val": ["episode-1"], "test": ["episode-2"], "holdout": []}
    if reverse:
        splits["train"], splits["test"] = splits["test"], splits["train"]
    store.freeze_split_assignments(splits, provenance={"fixture": True}, heldout_combinations=())
    return store


def test_assembly_preserves_source_membership_signals_and_immutable_output(tmp_path):
    sources = [corpus(tmp_path / "a"), corpus(tmp_path / "b", prefix="b", reverse=True)]
    original_hashes = [store.manifest_hash for store in sources]
    combined = assembly.assemble([store.root for store in sources], tmp_path / "combined")
    assert len(combined.episode_ids) == 6
    normalization = combined.manifest["normalization"]
    assert normalization["fit_split"] == "train"
    assert normalization["episode_ids"] == combined.manifest["splits"]["train"]
    for source in sources:
        for split, ids in source.manifest["splits"].items():
            for original_id in ids:
                combined_id = f"{source.manifest_hash}:{original_id}"
                assert combined_id in combined.manifest["splits"][split]
                original, copied = (
                    source.read_episode(original_id),
                    combined.read_episode(combined_id),
                )
                assert copied.session_id == original.session_id
                np.testing.assert_array_equal(copied.actions, original.actions)
                np.testing.assert_array_equal(
                    copied.observations["head"], original.observations["head"]
                )
                assert copied.metadata["assembly_lineage"]["episode_id"] == original_id
    assert [store.manifest_hash for store in sources] == original_hashes
    with pytest.raises(FileExistsError):
        assembly.assemble([sources[0].root], combined.root)
    with pytest.raises(ContractError, match="sealed"):
        combined.write_episode(deterministic_fixture("new"))


def test_assembly_rejects_cross_source_session_leakage_duplicate_and_apple(tmp_path):
    a = corpus(tmp_path / "a")
    b = corpus(tmp_path / "b", reverse=True)
    with pytest.raises(ContractError, match="crosses split"):
        assembly.assemble([a.root, b.root], tmp_path / "conflict")
    with pytest.raises(ContractError, match="duplicate"):
        assembly.assemble([a.root, a.root], tmp_path / "duplicate")
    apple = corpus(tmp_path / "apple", apple=True)
    with pytest.raises(ContractError, match="Apple"):
        assembly.assemble([apple.root], tmp_path / "forbidden")


def test_orchestrator_records_failure_and_all_six_fixed_runs(tmp_path, monkeypatch):
    source = corpus(tmp_path / "data")
    commands = []

    def supervise(command, log_path, budget):
        commands.append(command)
        checkpoint = Path(command[command.index("--output") + 1])
        status = "selection_failed" if len(commands) == 2 else "completed"
        checkpoint.with_suffix(".run.json").write_text(json.dumps({"status": status}))
        return {"returncode": 2 if status != "completed" else 0, "supervisor_timeout": False}

    monkeypatch.setattr(orchestration, "supervise", supervise)
    report = orchestration.run(source.root, tmp_path / "out", source.manifest_hash)
    assert report["status"] == "incomplete" and len(report["runs"]) == 6
    assert report["runs"][1]["status"] == "selection_failed"
    for command in commands:
        for name, expected in (
            ("steps", "3000"),
            ("batch-size", "16"),
            ("horizon", "4"),
            ("device", "cpu"),
            ("validation-every", "100"),
            ("selection", "noncollapsed_relative"),
        ):
            assert command[command.index("--" + name) + 1] == expected
    with pytest.raises(FileExistsError):
        orchestration.run(source.root, tmp_path / "out", source.manifest_hash)


def test_orchestrator_shared_budget_stops_later_runs(tmp_path, monkeypatch):
    source = corpus(tmp_path / "data")
    now = [0.0]
    clock_type = orchestration.RunClock
    monkeypatch.setattr(
        orchestration,
        "RunClock",
        lambda: clock_type(
            wall_clock=lambda: now[0],
            monotonic_clock=lambda: now[0],
            cpu_clock=lambda: 0,
        ),
    )

    def supervise(command, log_path, budget):
        now[0] += 600
        return {"returncode": -9, "supervisor_timeout": True}

    monkeypatch.setattr(orchestration, "supervise", supervise)
    report = orchestration.run(source.root, tmp_path / "out", source.manifest_hash)
    assert [row["max_seconds"] for row in report["runs"][:2]] == [600, 480]
    assert all(row["status"] == "not_started_total_budget" for row in report["runs"][2:])
    assert report["status"] == "incomplete"


def test_supervisor_kills_sleeping_child_and_records_nonzero_exit(tmp_path):
    result = orchestration.supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "timeout.log", 0.1
    )
    assert result["supervisor_timeout"] and result["returncode"] != 0
    assert result["supervisor_timing"]["elapsed_seconds"] < 3
    result = orchestration.supervise(
        [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failed.log", 3
    )
    assert result["returncode"] == 7 and not result["supervisor_timeout"]


def test_orchestrator_marks_source_change_invalid_and_stops(tmp_path, monkeypatch):
    source = corpus(tmp_path / "data")
    state = {"revision": "original"}
    monkeypatch.setattr(orchestration, "identity", lambda: dict(state))

    def supervise(command, log_path, budget):
        state["revision"] = "changed"
        return {"returncode": 0, "supervisor_timeout": False}

    monkeypatch.setattr(orchestration, "supervise", supervise)
    report = orchestration.run(source.root, tmp_path / "out", source.manifest_hash)
    assert report["runs"][0]["status"] == "invalid_integrity_changed"
    assert all(row["status"] == "not_started_integrity_changed" for row in report["runs"][1:])
