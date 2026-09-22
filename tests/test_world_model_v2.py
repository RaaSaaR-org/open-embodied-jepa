"""World model v2 runner/evaluator plumbing (NumPy only; no torch, no corpus).

A fake model whose dynamics are exact on a synthetic world checks that the offline
gate metrics reward correct action use and penalize the controls. None of this is a
learned result.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import readout_labels
from embodied_jepa import world_model_v2 as wm
from embodied_jepa.contracts import ContractError, RobotState, StateSchema

SCHEMA = StateSchema(("x", "y", "z"), ("m", "m", "m"), "fixture_v0")
GAIN = 0.01
CAMERA = "onboard_rgb"


class Latent:
    def __init__(self, values):
        self.values = values


class ExactFakeModel:
    """Encodes palm-minus-apple (stored in the fixture state) and integrates actions."""

    def encode(self, images, robot_state):
        assert isinstance(robot_state, RobotState) and CAMERA in images
        return Latent(np.array(robot_state.values[:, :3], np.float64))

    def predict(self, z, actions):
        assert actions.ndim == 4
        return Latent(z.values[:, None, None] + GAIN * np.cumsum(actions[..., 6:9], axis=2))

    def readout(self, z):
        values = z.values.astype(np.float32)
        shape = values.shape[:-1]
        return {
            "palm_minus_apple": values,
            "apple_height": np.zeros((*shape, 1), np.float32),
            "apple_minus_plate": np.zeros((*shape, 3), np.float32),
            "hand_contact": np.full((*shape, 1), 0.5, np.float32),
            "apple_held": np.full((*shape, 1), 0.5, np.float32),
        }

    def latent_statistics(self, latents):
        return {"collapsed_fraction": 0.0, "effective_rank": 5.0, "latent_std_mean": 1.0}


def _episode(rng, start, length):
    actions = rng.uniform(-1, 1, (length - 1, 14)).astype(np.float32)
    pma = np.empty((length, 3), np.float32)
    pma[0] = start
    for t in range(length - 1):
        pma[t + 1] = pma[t] + GAIN * actions[t, 6:9]
    return pma, actions


def fixture_arrays(seed=0, roots=3, root_length=70, branch_length=50, branch_frame=10):
    rng = np.random.default_rng(seed)
    episodes, rows = [], {}
    for r in range(roots):
        root = f"root-{r}"
        pma, actions = _episode(rng, rng.uniform(-0.2, 0.2, 3) + [0, 0, 0.3], root_length)
        episodes.append((root, pma, actions))
        rows[root] = {
            "episode_id": root,
            "metadata": {
                "kind": "root",
                "root_episode_id": root,
                "branch_frame_in_root": branch_frame,
                "privileged_outcome_labels": {"stages": {"grasp": bool(r % 2)}},
            },
        }
        for b in range(2):
            name = f"{root}-b{b}"
            bpma, bactions = _episode(rng, pma[branch_frame], branch_length)
            episodes.append((name, bpma, bactions))
            rows[name] = {
                "episode_id": name,
                "metadata": {
                    "kind": "branch",
                    "root_episode_id": root,
                    "branch_frame_in_root": branch_frame,
                    "privileged_outcome_labels": {"stages": {"grasp": bool(b)}},
                },
            }
    lengths = np.array([len(p) for _, p, _ in episodes], np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    total = int(lengths.sum())
    pma_all = np.concatenate([p for _, p, _ in episodes])
    actions_all = np.zeros((total, 14), np.float32)
    for (_, _, actions), offset in zip(episodes, offsets, strict=True):
        actions_all[offset : offset + len(actions)] = actions
    targets = {
        "palm_minus_apple": pma_all,
        "apple_height": np.zeros((total, 1), np.float32),
        "apple_minus_plate": np.zeros((total, 3), np.float32),
        "hand_contact": np.zeros((total, 1), np.float32),
        "apple_held": np.zeros((total, 1), np.float32),
    }
    return wm.EpisodeArrays(
        tuple(name for name, _, _ in episodes),
        offsets,
        lengths,
        np.zeros((total, 2, 2, 3), np.uint8),
        pma_all.copy(),
        np.ones((total, 3), bool),
        actions_all,
        np.arange(total, dtype=np.float64) * 0.05,
        targets,
        np.zeros(total, np.int16),
        rows,
    )


def test_readout_targets_from_privileged_labels():
    labels = {
        "privileged__palm_minus_apple_world": np.array([[0, 0, 0.1], [0, 0, 0.05]], np.float32),
        "privileged__apple_position_world": np.array([[0.3, 0.1, 0.77], [0.3, 0.1, 0.80]]),
        "privileged__plate_position_world": np.array([[0.5, 0.0, 0.75]] * 2),
        "privileged__hand_contact": np.array([True, True]),
    }
    result = readout_labels.targets(labels, 0.77)
    np.testing.assert_allclose(result["apple_height"][:, 0], [0.0, 0.03], atol=1e-6)
    np.testing.assert_allclose(result["apple_held"][:, 0], [0.0, 1.0])
    np.testing.assert_allclose(result["apple_minus_plate"][1], [-0.2, 0.1, 0.05], atol=1e-6)
    assert all(v.dtype == np.float32 and v.shape[0] == 2 for v in result.values())
    with pytest.raises(ContractError):
        readout_labels.targets(labels | {"privileged__hand_contact": np.array([True])}, 0.77)


def test_privileged_labels_require_acknowledgement(tmp_path):
    row = {"episode_id": "e", "metadata": {"training_labels": {"path": "labels/e.npz"}}}
    with pytest.raises(ContractError, match="acknowledge"):
        readout_labels.load_privileged(tmp_path, row)
    store = SimpleNamespace(manifest={"splits": {"train": ["a"], "val": ["b"], "test": ["c"]}})
    with pytest.raises(ContractError, match="only the train and val"):
        wm.load_split(store, "test", CAMERA, acknowledge_privileged_training_labels=True)
    with pytest.raises(ContractError, match="privileged"):
        wm.load_split(store, "val", CAMERA)


def test_rank_statistics():
    assert wm.auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)
    assert wm.auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert wm.auroc([0.1, 0.2], [1, 1]) is None
    assert wm.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert wm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_windows_batches_and_shuffled_control_stay_inside_episodes():
    arrays = fixture_arrays()
    windows = wm.window_starts(arrays, 16, stride=4)
    assert (windows[:, 1] + 16 <= arrays.lengths[windows[:, 0]] - 1).all()
    partner = wm.shuffled_pairing(windows)
    assert (windows[partner, 0] != windows[:, 0]).all()
    sequence, targets = wm.batch(arrays, windows[:5], 16, CAMERA, SCHEMA)
    assert sequence.actions.shape == (5, 16, 14)
    assert targets["palm_minus_apple"].shape == (5, 17, 3)
    first = arrays.offsets[windows[0, 0]] + windows[0, 1]
    np.testing.assert_array_equal(sequence.actions[0], arrays.actions[first : first + 16])


def test_exact_dynamics_pass_accuracy_and_action_controls():
    arrays = fixture_arrays()
    windows = wm.window_starts(arrays, 16, stride=4)
    metrics = wm.window_metrics(ExactFakeModel(), arrays, windows, CAMERA, SCHEMA)
    h8 = metrics["8"]
    assert h8["palm_apple_moving_windows"] > 10
    assert h8["palm_apple_moving_median_m"] < 1e-5
    assert h8["palm_apple_moving_persistence_median_m"] > 0.01
    assert h8["palm_apple_moving_shuffled_median_m"] > 0.01
    assert h8["approach_cost_median_abs_log_ratio"] < 1e-4
    siblings = wm.sibling_metrics(ExactFakeModel(), arrays, CAMERA, SCHEMA)
    assert siblings["start_state_max_label_mismatch_m"] == 0.0
    s16 = siblings["16"]
    assert s16["qualifying_ordered_pairs"] > 0
    assert s16["own_beats_swapped_fraction"] == 1.0
    assert s16["divergence_spearman"] == pytest.approx(1.0)


def test_action_blind_model_fails_sibling_and_shuffled_controls():
    class ActionBlind(ExactFakeModel):
        def predict(self, z, actions):
            return Latent(np.repeat(z.values[:, None, None], actions.shape[2], axis=2))

    arrays = fixture_arrays()
    windows = wm.window_starts(arrays, 16, stride=4)
    metrics = {
        "windows": wm.window_metrics(ActionBlind(), arrays, windows, CAMERA, SCHEMA),
        "siblings": wm.sibling_metrics(ActionBlind(), arrays, CAMERA, SCHEMA),
        "collapse": ActionBlind().latent_statistics(None),
    }
    gates = {
        "G1_palm_apple_moving_h8_median_m": 0.015,
        "G2_max_ratio_to_control": 0.8,
        "G3_apple_plate_h8_median_m": 0.02,
        "G4_apple_height_grasp_h8_median_abs_m": 0.01,
        "G5_held_auroc_min": 0.85,
        "G5_minimum_per_class": 20,
        "G6_max_cost_ratio": 1.5,
        "G7_minimum_pairs": 1,
        "G7a_own_beats_swapped_min": 0.7,
        "G7b_divergence_spearman_min": 0.5,
        "G8_max_collapsed_fraction": 0.05,
        "G8_min_effective_rank": 4.0,
        "G8_min_latent_std_mean": 0.1,
    }
    result = wm.evaluate_gates(metrics, gates)
    assert not result["all_passed"]
    assert not result["G2a_vs_persistence_h8"]["passed"]
    assert not result["G2b_vs_shuffled_actions_h8"]["passed"]
    assert not result["G7a_sibling_own_beats_swapped_h16"]["passed"]
    assert not result["G5_held_auroc_grasp_h8"]["passed"]  # one class only: not evaluable
    assert result["G8a_collapsed_fraction"]["passed"]
