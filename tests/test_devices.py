"""Device selection, the deterministic CUDA setup, and CUDA model/training checks.

Tests that need a GPU skip with "CUDA unavailable"; everything else runs on any host. These
are software checks: they make no learned-control claim.
"""

import json
import subprocess
import sys

import pytest

from embodied_jepa import devices


def test_importing_devices_does_not_import_torch():
    code = (
        "import sys, embodied_jepa.devices; "
        "assert 'torch' not in sys.modules, 'devices imported torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_resolve_rejects_unknown_names_and_auto_where_disallowed():
    with pytest.raises(devices.DeviceError, match="device must be one of"):
        devices.resolve("tpu")
    with pytest.raises(devices.DeviceError, match="device must be one of"):
        devices.resolve("auto", allow_auto=False)
    assert devices.resolve("cpu") == "cpu"
    assert not devices.available("tpu")


@pytest.mark.parametrize(
    "present,expected",
    [({"cuda", "mps"}, "cuda"), ({"mps"}, "mps"), (set(), "cpu"), ({"cuda"}, "cuda")],
)
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, present, expected):
    monkeypatch.setattr(devices, "available", lambda name: name == "cpu" or name in present)
    assert devices.resolve("auto") == expected


def test_require_reports_an_unavailable_device_with_the_callers_error(monkeypatch):
    monkeypatch.setattr(devices, "available", lambda name: name == "cpu")

    class CallerError(Exception):
        pass

    with pytest.raises(CallerError, match="'cuda' is unavailable"):
        devices.require("cuda", error=CallerError)
    with pytest.raises(devices.DeviceError, match="'mps' is unavailable"):
        devices.require("mps")


def test_config_auto_prefers_cuda(monkeypatch, tmp_path):
    yaml = pytest.importorskip("yaml")
    from embodied_jepa.config import ExperimentConfig

    (tmp_path / "data").mkdir()
    (tmp_path / "native.pt").touch()
    (tmp_path / "action.json").write_text(
        json.dumps(
            {
                "action_schema": "ee_delta_grasp_v0",
                "translation_per_step_m": 0.01,
                "rotation_per_step_rad": 0.05,
                "control_dt_s": 0.05,
                "joint_speed_limit_rad_s": 1,
            }
        )
    )
    raw = {
        "schema_version": 0,
        "runtime": {"device": "auto"},
        "world_model": {"backend": "native_jepa", "checkpoints": {"native_jepa": "native.pt"}},
        "embodiment": {
            "backend": "unitree_g1_dex3",
            "transport": "mujoco",
            "action_manifest": "action.json",
        },
        "dataset": {"root": "data"},
        "task": {"name": "reach"},
    }
    path = tmp_path / "auto.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(devices, "available", lambda name: True)
    assert ExperimentConfig.load(path).device == "cuda"
    monkeypatch.setattr(devices, "available", lambda name: name in ("cpu", "mps"))
    assert ExperimentConfig.load(path).device == "mps"
    raw["runtime"]["device"] = "cuda"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="'cuda' is unavailable"):
        ExperimentConfig.load(path)


def test_configure_determinism_is_a_noop_off_cuda(restore_determinism):
    before = devices.determinism_state()
    assert devices.configure_determinism("cpu") == before
    assert devices.configure_determinism("mps") == before


def test_configure_determinism_sets_every_flag(monkeypatch, restore_determinism):
    """The flags themselves need no GPU, so this runs everywhere."""
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    state = devices.configure_determinism("cuda")
    state.pop("sdp_backends")  # recorded, not changed
    assert state == {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }


def test_strict_determinism_turns_warnings_into_errors(monkeypatch, restore_determinism):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    state = devices.configure_determinism("cuda", strict=True)
    assert state["deterministic_algorithms"] and not state["deterministic_algorithms_warn_only"]


def test_configure_determinism_refuses_a_different_cublas_setting(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(devices.DeviceError, match="requires ':4096:8'"):
        devices.configure_determinism("cuda")


def test_configure_determinism_refuses_after_cuda_initialized_without_the_variable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(devices.DeviceError, match="before starting the process"):
        devices.configure_determinism("cuda")


def test_cpu_memory_report_is_empty_and_cpu_sync_is_a_noop():
    pytest.importorskip("torch")
    devices.synchronize("cpu")
    assert devices.memory_report("cpu") == {}
