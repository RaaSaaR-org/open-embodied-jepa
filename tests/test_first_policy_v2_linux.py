"""TASK-072 (``apple_first_policy_v2_linux``): the Linux replication of TASK-071.

The replication's runner and render check are TASK-071's files plus a declared list of textual
substitutions (below), so a reviewer can read the whole difference here. These tests pin that,
pin the manifest's frozen block to the module, check that only the declared keys of TASK-071's
frozen block changed, and check the declared reading. Software checks only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_linux as fpl

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2-linux.json").read_text()
)
V2_MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2.json").read_text()
)
HEADER_END = "from __future__ import annotations\n"

# (old, new) pairs, applied in order to the v2 file's code (everything from HEADER_END on). Each
# `old` must occur exactly once. The docstring above HEADER_END is the replication's own.
RUNNER_SUBSTITUTIONS = (
    (
        "from embodied_jepa import first_policy_v2 as fp2  # noqa: E402\n",
        "from embodied_jepa import devices  # noqa: E402\n"
        "from embodied_jepa import first_policy_v2 as fp2  # noqa: E402\n"
        "from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402\n",
    ),
    (
        'MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2.json"\n',
        'MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2-linux.json"\n'
        "fpl.configure_headless()  # MUJOCO_GL=egl before any MuJoCo import; workers inherit it\n",
    ),
    (
        '        "protocol": fp2.PROTOCOL,\n        "task": fp2.TASK,\n',
        '        "protocol": fpl.PROTOCOL,\n        "task": fpl.TASK,\n',
    ),
    (
        '        report["ended_utc"] = ',
        '        report["replication"] = fpl.read_replication(report)\n'
        '        report["ended_utc"] = ',
    ),
    (
        '    if manifest["frozen"] != fp2.frozen_block():\n',
        '    if manifest["frozen"] != fpl.frozen_block():\n',
    ),
    (
        "    mps = bool(torch.backends.mps.is_available())\n"
        '    device = "cpu" if smoke else fp2.TRAIN_DEVICE\n'
        "    if not smoke and not mps:\n"
        '        raise fp2.GuardError("G-device: MPS is not available")\n',
        '    report["platform"] = fpl.check_platform()  # G-platform\n'
        "    if not devices.available(fpl.TRAIN_DEVICE):\n"
        '        raise fp2.GuardError("G-device: CUDA is not available")\n'
        "    # Smokes train on CUDA too. Strict: an op without a deterministic kernel raises (V).\n"
        "    device = devices.require(fpl.TRAIN_DEVICE, strict=fpl.STRICT_DETERMINISM)\n",
    ),
    (
        '        "mps_available": mps,\n',
        '        "accelerator": devices.accelerator_info(device),\n'
        '        "determinism": devices.determinism_state(),\n',
    ),
    (
        '    workers = SMOKE["workers"] if smoke else fp2.SIM_WORKERS\n',
        '    workers = SMOKE["workers"] if smoke else fpl.SIM_WORKERS\n',
    ),
    (
        '    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])\n',
        '    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])\n'
        "    fpl.check_determinism(report)  # G-device at the end: still strict\n",
    ),
)
RENDER_SUBSTITUTIONS = (
    (
        "from embodied_jepa import first_policy_v2 as fp  # noqa: E402\n",
        "from embodied_jepa import first_policy_v2 as fp  # noqa: E402\n"
        "from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402\n",
    ),
    (
        '        "_run_first_policy_v2", ROOT / "scripts" / "run_first_policy_v2.py"\n',
        '        "_run_first_policy_v2_linux", ROOT / "scripts" / "run_first_policy_v2_linux.py"\n',
    ),
    (
        "    at_gated_count = workers == fp.SIM_WORKERS\n",
        "    at_gated_count = workers == fpl.SIM_WORKERS\n",
    ),
    (
        '        "check": "TASK-071 (TASK-067 R7): post-look re-render bit-identity, renders and '
        'workers",\n',
        '        "check": "TASK-072 (TASK-067 R7, on Linux): post-look re-render bit-identity",\n'
        '        "platform": fpl.check_platform(),\n',
    ),
    (
        '        "gated_worker_count": fp.SIM_WORKERS,\n',
        '        "gated_worker_count": fpl.SIM_WORKERS,\n',
    ),
    (
        '    parser.add_argument("--workers", type=int, default=fp.SIM_WORKERS)\n',
        '    parser.add_argument("--workers", type=int, default=fpl.SIM_WORKERS)\n',
    ),
)
PAIRS = (
    (
        "scripts/run_first_policy_v2.py",
        "scripts/run_first_policy_v2_linux.py",
        RUNNER_SUBSTITUTIONS,
    ),
    (
        "scripts/check_first_policy_v2_render.py",
        "scripts/check_first_policy_v2_linux_render.py",
        RENDER_SUBSTITUTIONS,
    ),
)


def derive(source: str, substitutions) -> str:
    code = source[source.index(HEADER_END) :]
    for old, new in substitutions:
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    return code


@pytest.mark.parametrize("original,copy,substitutions", PAIRS)
def test_copies_are_the_v2_files_plus_the_declared_substitutions(original, copy, substitutions):
    derived = derive((ROOT / original).read_text(), substitutions)
    text = (ROOT / copy).read_text()
    assert text[text.index(HEADER_END) :] == derived


def test_v2_files_are_untouched():
    for path, want in V2_MANIFEST["hashes"].items():
        if path.startswith(("src/", "scripts/", "configs/", "assets/", "docs/")):
            got = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            assert got == want, path


def test_manifest_frozen_block_matches_the_module():
    assert MANIFEST["frozen"] == fpl.frozen_block()


def test_only_the_declared_keys_differ_from_task071():
    mine, theirs = fpl.frozen_block(), fp2.frozen_block()
    changed = {k for k in mine.keys() | theirs.keys() if mine.get(k) != theirs.get(k)}
    assert changed == set(fpl.CHANGED_KEYS)
    assert {k: v for k, v in mine["training"].items() if k != "device"} == {
        k: v for k, v in theirs["training"].items() if k != "device"
    }
    assert mine["training"]["device"] == mine["devices"]["training"] == "cuda"
    assert mine["devices"]["features"] == mine["devices"]["rollouts"] == "cpu"
    assert mine["sim_workers"] == 16


def test_pins_carry_task071_and_add_the_new_files():
    pins = MANIFEST["hashes"]
    for path, want in V2_MANIFEST["hashes"].items():
        assert pins[path] == want, path
    for path in (
        "src/embodied_jepa/first_policy_v2_linux.py",
        "src/embodied_jepa/devices.py",
        "scripts/run_first_policy_v2_linux.py",
        "scripts/check_first_policy_v2_linux_render.py",
        "docs/experiments/apple_first_policy_v2_linux.md",
        "benchmarks/manifests/apple-first-policy-v2.json",
    ):
        assert path in pins, path
    for path, want in pins.items():
        got = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert got == want, path


def test_encoder_pins():
    assert MANIFEST["encoder_digests"]["pretrained"] == V2_MANIFEST["encoder_digests"]["pretrained"]
    assert MANIFEST["encoder_digests"]["floor"] != V2_MANIFEST["encoder_digests"]["floor"]
    assert MANIFEST["mujoco_version"] == V2_MANIFEST["mujoco_version"]


def test_run1_target_matches_the_task071_results_manifest():
    results = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2-results.json").read_text()
    )
    assert {a: c["success"] for a, c in results["m1_counts_of_16"].items()} == (
        fpl.RUN1["m1_success_of_16"]
    )
    assert results["outcome"] == fpl.RUN1["outcome"]
    assert results["decision"]["carried"] == fpl.RUN1["carried"]
    assert results["corpus"]["all"]["success"] == fpl.RUN1["corpus_counted_success_of_200"]


def _report(outcome, p3=16):
    counts = {arm: {"success": n} for arm, n in fpl.RUN1["m1_success_of_16"].items()}
    counts["P-3"] = {"success": p3}
    return {"smoke": False, "outcome": outcome, "stages": {"M1": {"counts": counts}}}


@pytest.mark.parametrize(
    "outcome,p3,row",
    [
        ("V", 16, "V"),
        (None, 16, "V"),
        ("CAL-ESCALATE", 16, "REP-EARLY-STOP"),
        ("S0-APPLE-FAIL", 16, "REP-EARLY-STOP"),
        ("S0-PLATE-FAIL", 16, "REP-EARLY-STOP"),
        ("M1-PASS", 16, "REPLICATED"),
        ("M1-PASS", 14, "REPLICATED"),
        ("M1-PASS", 13, "REPLICATED-M1-ONLY"),
        ("M1-PASS", 0, "REPLICATED-M1-ONLY"),
        ("M1-MOTOR-F-PARTIAL", 0, "NOT-REPLICATED"),
        ("M1-MOTOR-F-NONE", 0, "NOT-REPLICATED"),
        ("M1-PERCEPTION", 0, "NOT-REPLICATED"),
    ],
)
def test_declared_reading(outcome, p3, row):
    assert fpl.read_replication(_report(outcome, p3))["row"] == row


def test_reading_reports_signed_differences_and_skips_smokes():
    reading = fpl.read_replication(_report("M1-PASS", 15))
    assert reading["difference_vs_run1"]["P-3"] == -1
    assert reading["difference_vs_run1"]["C-3"] == 0
    assert fpl.read_replication({"smoke": True, "outcome": "V"})["row"] is None


def test_platform_guard(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    with pytest.raises(fpl.GuardError, match="G-platform"):
        fpl.check_platform()


def test_seed_ranges_are_task071s():
    block = fpl.frozen_block()
    assert block["seed_ranges"] == fp2.frozen_block()["seed_ranges"]
    assert block["cohort_D2"] == list(range(52000, 52016))


torch = pytest.importorskip("torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@needs_cuda
@pytest.mark.parametrize("heads", [1, len(fp2.EXPERT_PHASES)])
def test_policy_trains_strictly_and_bit_identically_on_cuda(heads, restore_determinism):
    """The owner's condition: the MLP policy (P/C/R, and F's per-phase heads) trains in strict
    deterministic mode on CUDA, with no nondeterminism warning, bit-identically twice."""
    import warnings

    import numpy as np

    from embodied_jepa import devices
    from embodied_jepa import first_policy_v2_model as fm2

    devices.require("cuda", strict=True)
    rng = np.random.default_rng(0)
    n, width = 2048, 132
    x = rng.normal(size=(n, width)).astype(np.float32)
    y = rng.normal(size=(n, 7)).astype(np.float32)
    steps = rng.integers(0, fp2.EXPERT_POLICY_STEPS, n)
    runs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            out = fm2.train(
                x,
                y,
                steps,
                x[:256],
                y[:256],
                steps[:256],
                heads=heads,
                device="cuda",
                updates=60,
                select_every=20,
                min_output_std=0.0,
            )
            runs.append(out)
    assert not devices.determinism_state()["deterministic_algorithms_warn_only"]
    assert not [w for w in caught if "deterministic" in str(w.message)]
    assert runs[0]["record"]["curve"] == runs[1]["record"]["curve"]
    for key, value in runs[0]["state"].items():
        assert torch.equal(value, runs[1]["state"][key]), key
