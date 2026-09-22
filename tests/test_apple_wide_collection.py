"""TASK-048 wide-jitter collector: plan, perturbation, labels and isolation checks.

Synthetic/unit checks only; they are never evidence of manipulation.
"""

import importlib.util
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import training_labels
from embodied_jepa.contracts import ContractError
from embodied_jepa.hand_crop import HandCrop
from embodied_jepa.scripted import apple_collector_policy

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load("collect_apple_wide")


def test_reset_rule_matches_task047_wide_reset():
    evaluator = load("evaluate_apple")
    for seed in (45000, 45007, 48000, 48199, 48931):
        expected = evaluator.wide_reset(seed)
        got = collector.wide_reset(seed)
        assert got["object_xy"] == expected["object_xy"]
        assert got["plate_xy"] == expected["plate_xy"]


def test_frozen_seed_range_is_new_and_split_is_whole_reset():
    seeds = set(collector.FROZEN_SEEDS)
    assert len(seeds) == 200 and not seeds & set(collector.PILOT_SEEDS)
    for used in (
        range(41000, 41102),  # mechanics probes
        range(42000, 42032),  # earlier TRAIN/VAL/TEST collection
        range(43000, 43005),  # narrow development
        range(44000, 44020),  # final cohort
        range(45000, 45008),  # wide development
    ):
        assert not seeds & set(used)
    plan = collector.make_plan(collector.FROZEN_SEEDS)
    assert plan == collector.make_plan(collector.FROZEN_SEEDS)
    assert Counter(r["split"] for r in plan["roots"]) == {"train": 170, "val": 20, "test": 10}
    ids = [r["episode_id"] for r in plan["roots"]]
    ids += [b["episode_id"] for r in plan["roots"] for b in r["branches"]]
    assert len(ids) == len(set(ids)) == 800
    kinds = Counter(b["kind"] for r in plan["roots"] for b in r["branches"])
    assert kinds == {kind: 120 for kind in collector.BRANCH_KINDS}
    assert Counter(r["noise_level"] for r in plan["roots"]) == {0: 50, 1: 50, 2: 50, 3: 50}
    assert sum(r["aim_offset_xy_m"] is not None for r in plan["roots"]) == 40
    for root in plan["roots"]:
        assert root["session_id"] == f"wide-reset-{root['seed']}"
        if root["aim_offset_xy_m"] is not None:
            assert 0.015 <= np.hypot(*root["aim_offset_xy_m"]) <= 0.03
        noise = {b["noise_seed"] for b in root["branches"]} | {root["noise_seed"]}
        assert len(noise) == 1 + collector.BRANCHES_PER_ROOT


def test_perturber_is_seeded_and_level_zero_is_the_script():
    base = np.zeros(14, np.float32)
    base[12:] = (-1, 1)
    action, kind = collector.Perturber(0, 1)(base)
    assert kind == "none" and np.array_equal(action, base)
    a, b = collector.Perturber(3, 7), collector.Perturber(3, 7)
    rows = [a(base) for _ in range(200)]
    assert all(np.array_equal(x[0], b(base)[0]) for x in rows)
    kinds = Counter(kind for _, kind in rows)
    assert kinds["ou"] and kinds["burst"]
    stacked = np.array([x[0] for x in rows])
    assert not stacked[:, :6].any() and stacked[:, 12].tolist() == [-1] * 200
    clipped = np.clip(stacked, collector.LOWER, collector.UPPER)
    assert (clipped[:, 6:12] <= 0.5).all() and (clipped[:, 6:12] >= -0.5).all()


def truth():
    return {
        "position_frame": "world",
        "base_position_world": np.zeros(3),
        "base_rotation_world": np.eye(3),
        "object_position": np.array([0.34, -0.18, 0.77]),
        "plate_position": np.array([0.49, -0.09, 0.745]),
        "container_surface_z": 0.745,
        "object_support_height": 0.027,
    }


def test_branch_modifications_touch_only_their_declared_phases():
    reference = apple_collector_policy(truth())
    for kind, params in (
        ("shift_close", {"offset_xy_m": [0.01, -0.02]}),
        ("early_lift", {"close_commands": 7}),
        ("noise_only", {}),
    ):
        controller = collector.Controller(
            apple_collector_policy(truth()), collector.Perturber(0, 0)
        )
        controller.apply_branch(kind, params, 5)
        assert controller.perturber.level == 1  # branches always get their own stream
        for index, (new, old) in enumerate(
            zip(controller.policy.phases, reference.phases, strict=True)
        ):
            moved = not np.allclose(new.target_base, old.target_base)
            assert moved == (kind == "shift_close" and index in (2, 3))
            assert (new.commands != old.commands) == (kind == "early_lift" and index == 2)
    shifted = apple_collector_policy(truth())
    collector.shifted(shifted, (0, 1, 2, 3), [0.02, 0.0])
    assert np.allclose(
        shifted.phases[1].target_base[:2] - reference.phases[1].target_base[:2], [0.02, 0.0]
    )
    assert np.allclose(shifted.phases[4].target_base, reference.phases[4].target_base)


class FakeRobot:
    def __init__(self):
        self.manifest = {"translation_per_step_m": 0.015, "rotation_per_step_rad": 0.06}

    def ee_pose(self, side):
        return np.array([0.3, -0.2, 0.9]), np.eye(3)


def test_grasp_overrides_follow_phase():
    robot = FakeRobot()
    controller = collector.Controller(apple_collector_policy(truth()), collector.Perturber(0, 0))
    controller.apply_branch("weak_close", {"grasp_target": 0.3}, 1)
    controller.perturber = collector.Perturber(0, 0)
    controller.policy.phase_index = 2
    assert controller.command(robot)[1][13] == pytest.approx(0.3)
    controller.policy.phase_index = 4
    assert controller.command(robot)[1][13] == pytest.approx(1.0)
    opener = collector.Controller(apple_collector_policy(truth()), collector.Perturber(0, 0))
    opener.apply_branch("open_during_lift", {"open_after_lift_commands": 5}, 1)
    opener.perturber = collector.Perturber(0, 0)
    opener.policy.phase_index, opener.policy.phase_step = 3, 4
    assert opener.command(robot)[1][13] == 1.0
    opener.policy.phase_step = 5
    assert opener.command(robot)[1][13] == -1.0


def test_label_sidecars_gate_privileged_groups_and_verify_hashes(tmp_path):
    labels = {
        "robot__hand_crop_window": np.zeros((3, 2), np.int16),
        "collector__phase_index": np.zeros(2, np.int8),
        "privileged__apple_position_world": np.ones((3, 3), np.float32),
    }
    with pytest.raises(ContractError, match="prefixed"):
        training_labels.encode({"apple_position": np.ones(3)})
    with pytest.raises(ContractError, match="finite"):
        training_labels.encode({"privileged__x": np.array([np.nan])})
    reference = training_labels.write(tmp_path, "ep-0", labels)
    with pytest.raises(FileExistsError):
        training_labels.write(tmp_path, "ep-0", labels)
    assert set(training_labels.load(tmp_path, reference)) == {"robot__hand_crop_window"}
    for groups in (("privileged",), ("collector",), ("robot", "privileged")):
        with pytest.raises(ContractError, match="acknowledge"):
            training_labels.load(tmp_path, reference, groups=groups)
    full = training_labels.load(
        tmp_path,
        reference,
        groups=training_labels.GROUPS,
        acknowledge_privileged_training_labels=True,
    )
    assert set(full) == set(labels)
    (tmp_path / reference["path"]).write_bytes(
        training_labels.encode(
            labels | {"privileged__apple_position_world": np.zeros((3, 3), np.float32)}
        )[0]
    )
    with pytest.raises(ContractError, match="hash mismatch"):
        training_labels.load(tmp_path, reference)
    with pytest.raises(ContractError, match="path"):
        training_labels.load(tmp_path, reference | {"path": "../x.npz"})


def test_hand_crop_window_uses_only_palm_and_camera_link_poses():
    crop = object.__new__(HandCrop)
    crop.render_size, crop.crop_size, crop.camera_id = 320, 112, 0
    crop.focal_px = 160 / np.tan(np.deg2rad(75) / 2)

    def data(palm):
        return SimpleNamespace(
            site=lambda name: SimpleNamespace(xpos=np.array(palm, float)),
            cam_xpos=np.zeros((1, 3)),
            cam_xmat=np.eye(3).reshape(1, 9),
        )

    assert crop.window(data([0, 0, -1])) == (104, 104)  # on the optical axis: centred
    top, left = crop.window(data([0.2, 0.0, -1]))
    assert top == 104 and left > 104
    assert crop.window(data([5, 5, -1])) == (0, 208)  # clamped inside the image
    assert crop.window(data([0, 0, 1])) == (104, 104)  # behind the camera


def test_guard_refusal_text_matches_the_ceiling_convention():
    from embodied_jepa.object_ceiling import GUARD_REFUSALS

    assert (collector.GUARD_REFUSAL,) == GUARD_REFUSALS


def test_model_planner_and_evaluation_code_never_import_training_labels():
    package = ROOT / "src/embodied_jepa"
    allowed = {"training_labels.py"}
    for path in package.rglob("*.py"):
        if path.name in allowed:
            continue
        assert not re.search(r"training_labels", path.read_text()), path
    for script in ("evaluate_apple.py", "train_apple_sensor.py", "evaluate_mvp.py"):
        assert "training_labels" not in (ROOT / "scripts" / script).read_text()


def test_branch_kinds_are_not_confounded_with_aim_offset_and_plan_hash_is_pinned():
    import hashlib
    import json

    plan = collector.make_plan(collector.FROZEN_SEEDS)
    aimed = Counter(
        b["kind"] for r in plan["roots"] if r["aim_offset_xy_m"] is not None for b in r["branches"]
    )
    assert aimed == {kind: 24 for kind in collector.BRANCH_KINDS}
    import platform
    import sys

    def rounded(value):
        if isinstance(value, float):
            return round(value, 9)
        if isinstance(value, list):
            return [rounded(item) for item in value]
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        return value

    def sha(value):
        return hashlib.sha256((json.dumps(value, indent=2) + "\n").encode()).hexdigest()

    # Platform-robust pin: sin/cos of the offset draws can differ in the last ulp across
    # CPU architectures, so every platform checks the plan rounded to 1e-9.
    assert sha(rounded(plan)) == "b88579ad69ebeb4f9d6cc7d1b7087eb2c0833cc7dc3e75a44bf392efa1f8ec98"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        # Exact bytes of the frozen plan, generated and executed on macOS arm64.
        assert sha(plan) == "15ed1a99e45114a5cec6013d345804ec561fad859dc3f0dd89dd93ec1e33062c"
