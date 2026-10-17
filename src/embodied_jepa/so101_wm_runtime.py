"""TASK-100's runner helpers shared by the stage scripts (report header, guards, model building
and loading, chunked roll-outs). NumPy at import; torch is imported inside the functions that
need it. Runners import this module, never another runner script.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embodied_jepa import run_tools as rt
from embodied_jepa import so101_wm as sw
from embodied_jepa import so101_wm_data as sd

ROOT = Path(__file__).resolve().parents[2]
STATE_SCHEMA_NAMES = ("features.placeholder",)


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def tree_dirty() -> bool:
    return bool(git("status", "--porcelain", "--untracked-files=no"))


def header(kind: str, **extra) -> dict:
    report = {
        "task": sw.TASK,
        "protocol": sw.PROTOCOL,
        "kind": kind,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": tree_dirty(),
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    } | extra
    rt.assert_local_import(ROOT, report)
    return report


def require_clean(report: dict, strict: bool) -> None:
    """§12: the tracked tree is clean at every gated stage's start and end."""
    if strict and tree_dirty():
        raise rt.GuardError("G-tree: the tracked tree is dirty")
    report["tracked_tree_dirty_checked"] = bool(strict)
    check_frozen(report, strict)


def write_json(path, payload) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=1, default=_default))
    return sd.sha256_file(path)


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


# ----- feature stores ----------------------------------------------------------------------------
def load_fit(folder, expected_report_sha256: str | None = None) -> tuple[sd.Moments, dict]:
    """The fit set's moments of a feature folder (``report_fit.json``), hash-checked."""
    folder = Path(folder)
    path = folder / "report_fit.json"
    got = sd.sha256_file(path)
    if expected_report_sha256 is not None and got != expected_report_sha256:
        raise rt.GuardError("G-features: report_fit.json differs from the recorded one")
    fit = json.loads(path.read_text())
    moments = sd.Moments.load(folder / "moments.npz", fit["moments_sha256"])
    return moments, fit | {"report_sha256": got}


def verify_features(folder, splits) -> dict:
    """§12: re-hash a feature store's files against its feature report(s)."""
    folder = Path(folder)
    out = {}
    for name in ("report.json", "report_test.json"):
        path = folder / name
        if not path.exists():
            continue
        files = json.loads(path.read_text())["files_sha256"]
        mine = [s for s in splits if s in files]
        out |= sd.verify_store_files(folder, files, mine)
    missing = [s for s in splits if not any(k.endswith(f"_{s}.json") for k in out)]
    if missing:
        raise rt.GuardError(f"G-features: no feature report covers {missing}")
    return out


# ----- models ------------------------------------------------------------------------------------
def state_schema():
    from embodied_jepa.contracts import StateSchema

    return StateSchema(STATE_SCHEMA_NAMES, ("1",), "task100_cached_features_v0")


def build_model(arm: str, seed: int, grid: str, metadata: dict, device: str):
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
    from embodied_jepa.models import schema_frozen_tokens as sft

    spec = sw.check_arm(arm)
    return sft.create(
        spec["backend"],
        state_schema=state_schema(),
        action_schema=JOINT_DELTA_SO101_V0,
        config=sw.model_config(arm, grid),
        device=device,
        seed=int(seed),
        metadata=metadata,
    )


def load_job(job_dir, device: str, *, which: str = "model.pt"):
    """A finished training job's model (kept checkpoint by default) and report; refuses another
    implementation hash than the job's and the current code's."""
    job_dir = Path(job_dir)
    report = json.loads((job_dir / "report.json").read_text())
    model = build_model(report["arm"], report["seed"], report["grid"], report["metadata"], device)
    if model.implementation_sha256 != report["implementation_sha256"]:
        raise rt.GuardError(f"G-checkpoint: {job_dir.name} was written by other code")
    path = job_dir / which
    if which == "model.pt" and sd.sha256_file(path) != report["checkpoint_sha256"]:
        raise rt.GuardError(f"G-checkpoint: {job_dir.name}/model.pt differs from its report")
    model.load(path)
    return model, report


def command_set(commands, arm: str) -> np.ndarray:
    """N arms see zero commands in training and evaluation (§4)."""
    return np.zeros_like(commands) if sw.check_arm(arm)["zero_actions"] else commands


def rollout_chunks(model, start, commands, horizons, *, chunk: int = 512):
    """Yield (lo, {h: float32 [n, D]}) chunk by chunk (bounded host memory)."""
    commands = np.asarray(commands, np.float32)
    for lo in range(0, len(start), chunk):
        yield lo, model.predict_at(start[lo : lo + chunk], commands[lo : lo + chunk], horizons)


def criterion(model, view: sd.FeatureView, arm: str, *, chunk: int = 512) -> float:
    """§5's val criterion: the mean latent error of the recursive roll-out at steps 1-16 from
    every root of ``view`` under its executed commands (N arms: zero commands)."""
    starts, _ = view.roots()
    steps = tuple(range(1, sw.CRITERION_STEPS + 1))
    s = view.store
    total, count = 0.0, 0
    for lo in range(0, len(starts), chunk):
        idx = starts[lo : lo + chunk]
        start = s.standardised(idx)
        commands = command_set(s.actions[idx[:, None] + np.arange(sw.CRITERION_STEPS)], arm)
        pred = model.predict_features(start, commands)
        targets = s.standardised(idx[:, None] + np.arange(1, sw.CRITERION_STEPS + 1))
        d = pred.astype(np.float64) - targets
        total += float((d * d).mean(axis=2).sum())
        count += len(idx) * len(steps)
    value = total / count
    if not np.isfinite(value):
        raise rt.GuardError("G-finite: non-finite val criterion")
    return value


# ----- the freeze (§10.7, §12) -------------------------------------------------------------------
MANIFEST = ROOT / "benchmarks" / "manifests" / "so101-world-model-v1.json"


def frozen_files_check(root=ROOT, manifest=MANIFEST) -> dict:
    """``{path: bool}`` for every file the manifest pins (own code and composed files)."""
    import hashlib

    pins = json.loads(Path(manifest).read_text())
    files = dict(pins["files_sha256"]) | dict(pins["composed_files_sha256"])
    return {
        rel: (Path(root) / rel).exists()
        and hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() == pin
        for rel, pin in files.items()
    }


def check_frozen(report: dict, strict: bool) -> None:
    """Gated stages refuse unless every pinned file matches the frozen manifest (G-frozen)."""
    if not strict:
        return
    if not MANIFEST.exists():
        raise rt.GuardError("G-frozen: no frozen manifest")
    status = json.loads(MANIFEST.read_text()).get("status")
    check = frozen_files_check()
    report["frozen_manifest"] = {"status": status, "sha256": sd.sha256_file(MANIFEST)}
    if status != "FROZEN" or not all(check.values()):
        bad = [k for k, v in check.items() if not v]
        raise rt.GuardError(f"G-frozen: status {status}, files differing from their pins: {bad}")
