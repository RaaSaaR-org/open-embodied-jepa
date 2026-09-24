"""Guards for the behaviour-cloning policy and trainer (TASK-056).

These tests cover the things that would silently invalidate the protocol's comparison rather
than announce themselves: the A0 ablation being made dishonest by fusing proprioception into
the image pathway, the exclusion rules drifting from the frozen 137/15/8 table, a reloaded
policy normalizing with the identity, and a collapsed head being selected because its action
error happens to look small.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa.contracts import ACTION_DIM, ACTION_NAMES, ContractError, validate_actions

torch = pytest.importorskip("torch")

from embodied_jepa import cloning  # noqa: E402
from embodied_jepa.contracts import RobotState, StateSchema  # noqa: E402
from embodied_jepa.models.base import VisualModel  # noqa: E402
from embodied_jepa.policy import (  # noqa: E402
    FREE_ACTION_INDICES,
    PINNED_ACTION_VALUES,
    ClonedPolicy,
    NoEncoder,
    _check_pinned_bounds,
    normalize_state,
    state_scale_from_moments,
)
from embodied_jepa.registry import POLICIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-policy-v1.json"


def schema(dimension=6):
    return StateSchema(
        tuple(f"j{i}.position" for i in range(dimension)),
        ("rad",) * dimension,
        "test_proprio_v0",
    )


def policy(dimension=6, seed=0):
    state_schema = schema(dimension)
    p = ClonedPolicy(state_schema, NoEncoder(), device="cpu", seed=seed)
    p.fit_state_normalization(
        np.zeros(dimension, np.float32),
        np.ones(dimension, np.float32),
        training_episode_ids=("a", "b"),
    )
    return p


def test_policy_is_registered_and_lazy():
    POLICIES.require("cloned_bc_v1")


def test_free_and_pinned_components_partition_the_action():
    assert set(FREE_ACTION_INDICES) | set(PINNED_ACTION_VALUES) == set(range(ACTION_DIM))
    assert not set(FREE_ACTION_INDICES) & set(PINNED_ACTION_VALUES)
    # The names must be the right ones: the right arm and the right grasp, nothing else.
    assert [ACTION_NAMES[i] for i in FREE_ACTION_INDICES] == [
        "right_dx",
        "right_dy",
        "right_dz",
        "right_droll",
        "right_dpitch",
        "right_dyaw",
        "right_grasp",
    ]


def test_act_emits_a_contract_valid_action_with_the_pinned_components_pinned():
    p = policy()
    state = RobotState(
        np.zeros((1, 6), np.float32),
        np.ones((1, 6), bool),
        np.array([1.0]),
        p.state_schema,
    )
    action = p.act({}, state)
    assert action.shape == (ACTION_DIM,) and action.dtype == np.float32
    validate_actions(action[None, None, None], ndim=4)
    for index, value in PINNED_ACTION_VALUES.items():
        assert action[index] == value


def test_act_clips_into_the_contract_range():
    """A regression head is unbounded; validate_actions rejects anything outside [-1, 1]."""
    p = policy()
    with torch.no_grad():
        p.head[-1].bias.fill_(50.0)
    state = RobotState(
        np.zeros((1, 6), np.float32), np.ones((1, 6), bool), np.array([1.0]), p.state_schema
    )
    action = p.act({}, state)
    validate_actions(action[None, None, None], ndim=4)
    assert action[list(FREE_ACTION_INDICES)].max() == pytest.approx(1.0)


def test_pinned_bounds_check_rejects_bounds_that_free_a_pinned_component():
    lower = np.array([0.0] * 6 + [-0.5] * 6 + [-1.0, -1.0], np.float64)
    upper = np.array([0.0] * 6 + [0.5] * 6 + [-1.0, 1.0], np.float64)
    _check_pinned_bounds(lower, upper)  # the protocol's frozen bounds
    freed = upper.copy()
    freed[12] = 1.0  # left_grasp no longer pinned
    with pytest.raises(ContractError, match="left_grasp"):
        _check_pinned_bounds(lower, freed)


def test_state_scaling_matches_the_world_model_exactly():
    """A0 has no encoder to borrow buffers from, so the scaling is recomputed. It must agree.

    The constants are read off ``VisualModel`` rather than copied; this asserts the arithmetic
    agrees too, so the A0 arm and the encoder arms cannot normalize differently.
    """
    rng = np.random.default_rng(0)
    dimension = 12
    mean = rng.normal(size=dimension)
    std = np.abs(rng.normal(size=dimension)) * 0.1
    std[0] = 1e-6  # below STATE_NOISE_STD: treated as constant
    std[1] = 1e-3  # between noise and floor: floored
    _, scale = state_scale_from_moments(mean, std)
    constant = std < VisualModel.STATE_NOISE_STD
    expected = np.where(
        constant, VisualModel.STATE_CONSTANT_SCALE, np.maximum(std, VisualModel.STATE_FLOOR_STD)
    )
    assert scale == pytest.approx(expected)
    assert scale[0] == VisualModel.STATE_CONSTANT_SCALE
    assert scale[1] == VisualModel.STATE_FLOOR_STD


def test_state_normalization_clips_and_zeroes_masked_dimensions():
    mean = np.zeros(3, np.float32)
    scale = np.ones(3, np.float32)
    values = np.array([[1000.0, 2.0, 3.0]], np.float32)
    mask = np.array([[True, True, False]])
    out = normalize_state(values, mask, mean, scale)
    assert out[0, 0] == VisualModel.STATE_CLIP
    assert out[0, 1] == 2.0
    assert out[0, 2] == 0.0


def test_checkpoint_round_trips_the_normalization_buffers(tmp_path):
    """The buffers live on the module, not the head. Losing them normalizes with the identity."""
    p = policy()
    with torch.no_grad():
        p.state_mean.fill_(3.0)
        p.state_scale.fill_(7.0)
    path = tmp_path / "policy.pt"
    p.save(path)
    with pytest.raises(FileExistsError):
        p.save(path)

    reloaded = ClonedPolicy(p.state_schema, NoEncoder(), device="cpu", seed=0)
    reloaded.load(path)
    assert float(reloaded.state_mean[0]) == 3.0
    assert float(reloaded.state_scale[0]) == 7.0
    assert bool(reloaded.state_normalization_fitted)

    state = RobotState(
        np.ones((1, 6), np.float32), np.ones((1, 6), bool), np.array([1.0]), p.state_schema
    )
    assert reloaded.predict_free({}, state) == pytest.approx(p.predict_free({}, state))


def test_an_unfitted_policy_refuses_to_act():
    p = ClonedPolicy(schema(), NoEncoder(), device="cpu", seed=0)
    state = RobotState(
        np.zeros((1, 6), np.float32), np.ones((1, 6), bool), np.array([1.0]), p.state_schema
    )
    with pytest.raises(ContractError, match="normalization"):
        p.act({}, state)


def test_normalization_cannot_be_refit():
    p = policy()
    with pytest.raises(ContractError, match="frozen"):
        p.fit_state_normalization(
            np.zeros(6, np.float32), np.ones(6, np.float32), training_episode_ids=("c",)
        )


def test_a_collapsed_head_is_never_selected():
    """Eligibility exists because a constant head can post a small median action error."""
    eligibility = {"min_output_std": 0.02}
    collapsed = {"median_abs_error": 0.001, "output_std_min": 0.0}
    healthy = {"median_abs_error": 0.5, "output_std_min": 0.3}
    assert cloning.selectable(100, collapsed, None, eligibility) is False
    assert cloning.selectable(100, healthy, None, eligibility) is True
    # Even against a worse incumbent, a collapsed head does not win.
    assert cloning.selectable(100, collapsed, (10, 0.9), eligibility) is False


def test_the_untrained_step_zero_policy_is_never_selectable():
    """A random head passes the collapse rule -- a LayerNorm-MLP at init has healthy std.

    Without a step guard it is written as ``best``, never beaten, and the run still reports
    ``completed``, feeding the gates a policy that learned nothing. ``world_model_v2`` guards
    the same way for the same reason.
    """
    eligibility = {"min_output_std": 0.02}
    healthy = {"median_abs_error": 0.4, "output_std_min": 0.3}
    assert cloning.selectable(0, healthy, None, eligibility) is False
    assert cloning.selectable(1, healthy, None, eligibility) is True


def test_surviving_root_rule_excludes_branches_and_aim_offset_roots():
    assert cloning.is_surviving_root("wide-48000") is True
    assert cloning.is_surviving_root("wide-48004") is False  # index 4: aim offset
    assert cloning.is_surviving_root("wide-48000-b0-shift_close") is False
    assert cloning.is_surviving_root("wide-48004-b1-weak_close") is False
    assert cloning.is_surviving_root("wide-49000") is False  # outside the corpus seed range


def test_surviving_counts_match_the_protocol_and_the_manifest():
    """The trainer asserts these at run time; this asserts the constant it asserts against."""
    order = list(cloning.FROZEN_SEEDS)
    np.random.default_rng(48).shuffle(order)
    n_val, n_test = max(1, round(0.10 * len(order))), max(1, round(0.05 * len(order)))
    split = {
        seed: ("val" if i < n_val else "test" if i < n_val + n_test else "train")
        for i, seed in enumerate(order)
    }
    counts = {
        name: sum(
            1
            for s in cloning.FROZEN_SEEDS
            if split[s] == name and cloning.is_surviving_root(f"wide-{s}")
        )
        for name in ("train", "val", "test")
    }
    assert counts == {"train": 137, "val": 15, "test": 8}
    assert cloning.SURVIVING_ROOTS == {"train": 137, "val": 15}
    declared = json.loads(MANIFEST.read_text())["exclusions"]["surviving_roots"]
    assert (declared["train"], declared["val"], declared["test"]) == (137, 15, 8)


def test_the_trainer_never_decodes_the_test_split():
    """Structural: load_split refuses anything but train and val, and the constant has no test."""
    assert "test" not in cloning.SURVIVING_ROOTS
    source = (ROOT / "src" / "embodied_jepa" / "cloning.py").read_text()
    assert '"test"' not in source.split("def load_bc_split")[1].split("def ")[0]


def test_the_image_pathway_uses_image_features_not_encode():
    """A2 must not consume ``encode``: E0 has state_fusion, which would fuse proprioception in.

    If that happened, the A0 ablation would no longer isolate vision and gate G2 would be
    measuring nothing. Structural check, because the failure is silent.
    """
    import inspect

    from embodied_jepa.policy import FrozenEncoder

    body = inspect.getsource(FrozenEncoder.features)
    assert "image_features" in body
    assert "self.model.encode(" not in body
    # And the whole module must never reach for the fused encoder on a model.
    source = (ROOT / "src" / "embodied_jepa" / "policy.py").read_text()
    assert "model.encode(" not in source


class MovingEncoder:
    """A trainable stand-in whose features shift once its parameter is updated."""

    kind = "moving"
    feature_dim = 4
    trainable = True

    def __init__(self):
        self.model = torch.nn.Linear(4, 4)
        self.calls = 0

    def features(self, images, *, inference: bool = False):
        self.calls += 1
        self.last_inference = inference
        batch = int(images["onboard_rgb"].shape[0])
        base = torch.ones(batch, 4)
        return self.model(base)

    def parameters(self):
        return self.model.parameters()

    def weights_sha256(self):
        import hashlib

        digest = hashlib.sha256()
        state = self.model.state_dict()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def provenance(self):
        # Carries the digest, like the real FrozenEncoder. Without it the test named for the
        # trainable save/load path is structurally incapable of seeing the field, which is how
        # the A3-unloadable regression stayed green.
        return {"kind": self.kind, "frozen": False, "model_weights_sha256": self.weights_sha256()}


def test_a_trainable_encoder_is_saved_and_restored(tmp_path):
    """B2: a plain-object feature source is not in state_dict() and was silently discarded.

    For A3 the encoder IS part of the trained artifact. Losing it pairs the reloaded head with
    the original E0 encoder, and the arm cannot be reproduced.
    """
    source = MovingEncoder()
    p = ClonedPolicy(schema(), source, device="cpu", seed=0)
    p.fit_state_normalization(
        np.zeros(6, np.float32), np.ones(6, np.float32), training_episode_ids=("a",)
    )
    with torch.no_grad():
        source.model.weight.fill_(2.5)
    path = tmp_path / "a3.pt"
    p.save(path)
    saved = torch.load(path, weights_only=True)
    assert saved["feature_source_weights"] is not None

    # The reloading policy starts from DIFFERENT encoder weights, as A3 does in practice: the
    # saved digest is the post-training encoder, the live one is a fresh initialization.
    fresh = MovingEncoder()
    with torch.no_grad():
        fresh.model.weight.fill_(0.125)
    assert fresh.weights_sha256() != source.weights_sha256()
    q = ClonedPolicy(schema(), fresh, device="cpu", seed=0)
    q.load(path)
    assert float(fresh.model.weight[0, 0].detach()) == 2.5
    # After the restore the digest must match: "the encoder in memory is byte-identical to the
    # one that trained this head". Comparing it BEFORE the restore made A3 unloadable.
    assert fresh.weights_sha256() == saved["feature_source"]["model_weights_sha256"]


def test_a_tampered_encoder_checkpoint_is_refused(tmp_path):
    """The post-restore digest is a real round-trip assertion, not a formality."""
    source = MovingEncoder()
    p = ClonedPolicy(schema(), source, device="cpu", seed=0)
    p.fit_state_normalization(
        np.zeros(6, np.float32), np.ones(6, np.float32), training_episode_ids=("a",)
    )
    path = tmp_path / "a3.pt"
    p.save(path)
    payload = torch.load(path, weights_only=True)
    payload["feature_source_weights"]["weight"] = torch.zeros(4, 4)
    torch.save(payload, path)

    q = ClonedPolicy(schema(), MovingEncoder(), device="cpu", seed=0)
    with pytest.raises(ContractError, match="did not reproduce the trained encoder"):
        q.load(path)


def test_a_frozen_arms_checkpoint_records_no_encoder_weights(tmp_path):
    p = policy()
    path = tmp_path / "a0.pt"
    p.save(path)
    assert torch.load(path, weights_only=True)["feature_source_weights"] is None


def test_a_checkpoint_refuses_a_different_feature_source(tmp_path):
    """An A2 head on an A1 encoder is the cross-arm confusion gate G2 exists to detect."""
    p = policy()
    path = tmp_path / "a0.pt"
    p.save(path)
    other = ClonedPolicy(schema(), MovingEncoder(), device="cpu", seed=0)
    with pytest.raises(ContractError, match="different feature source"):
        other.load(path)


def test_a_frozen_arm_refuses_a_checkpoint_from_a_differently_weighted_encoder(tmp_path):
    """The A1-versus-A2 cross-arm case, which is what the digest check is NAMED for.

    The existing refusal test pairs ``NoEncoder`` against ``MovingEncoder``, so it exits at the
    ``saved_source != live_source`` comparison and never reaches the digest. That left the
    frozen branch — two encoders of the same kind with different weights, i.e. exactly A1
    against A2 — exercised by nothing in the suite. This is the coverage gap an independent
    probe found; a test whose fixtures cannot reach the case under test is how three defects in
    this protocol stayed invisible.
    """

    def frozen_source(fill):
        source = MovingEncoder()
        source.trainable = False
        with torch.no_grad():
            # BOTH tensors: Linear's bias is randomly initialized per instance, so filling
            # only the weight leaves two "identical" encoders with different digests -- which
            # is the digest working correctly, and would look like a test failure.
            source.model.weight.fill_(fill)
            source.model.bias.fill_(fill)
        return source

    trained = frozen_source(0.75)
    p = ClonedPolicy(schema(), trained, device="cpu", seed=0)
    p.fit_state_normalization(
        np.zeros(6, np.float32), np.ones(6, np.float32), training_episode_ids=("a",)
    )
    path = tmp_path / "a2.pt"
    p.save(path)
    # Nothing is restored for a frozen arm, so the message must NOT claim a failed restore.
    other = ClonedPolicy(schema(), frozen_source(-0.25), device="cpu", seed=0)
    with pytest.raises(ContractError, match="different frozen encoder"):
        other.load(path)
    # The matching frozen encoder still loads.
    ClonedPolicy(schema(), frozen_source(0.75), device="cpu", seed=0).load(path)


def test_provenance_distinguishes_two_encoders_of_the_same_architecture():
    """The A1-versus-A2 case, which differing ``kind`` does NOT exercise.

    Every other field of ``provenance()`` is weight-independent: ``model_implementation_sha256``
    hashes source files and ``model_parameters`` is a count. So a randomly initialized encoder
    and a loaded checkpoint of the SAME architecture produce identical provenance unless the
    weights are digested. Without this the compatibility check in ``load`` would pass while
    claiming to prevent exactly the confusion it is named for.
    """
    from embodied_jepa.policy import FrozenEncoder

    class Tiny(torch.nn.Module):
        backend = "tiny"
        camera_names = ("onboard_rgb",)
        config = {"latent_dim": 4}

        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 4)

        implementation_sha256 = "identical-for-both"

    a1, a2 = Tiny(), Tiny()
    with torch.no_grad():
        a2.layer.weight.fill_(0.5)  # "loaded a checkpoint"

    # FrozenEncoder type-checks its argument, so exercise the digest directly.
    digest_a1 = FrozenEncoder.weights_sha256.__get__(type("S", (), {"model": a1})(), object)()
    digest_a2 = FrozenEncoder.weights_sha256.__get__(type("S", (), {"model": a2})(), object)()
    assert digest_a1 != digest_a2, (
        "two encoders of the same architecture with different weights must be distinguishable"
    )
    assert (
        digest_a1 == FrozenEncoder.weights_sha256.__get__(type("S", (), {"model": a1})(), object)()
    ), "the digest must be stable for unchanged weights"


def test_evaluate_policy_recomputes_features_when_the_encoder_is_trainable():
    """B1: a stale VAL cache scores a live head against features frozen at initialization.

    ``features=None`` must mean "recompute", not "no image pathway", whenever the source is
    trainable -- otherwise A3's val curve, best_step and selected checkpoint are meaningless
    while still looking like a merely-bad curve.
    """
    source = MovingEncoder()
    p = ClonedPolicy(schema(), source, device="cpu", seed=0)
    p.fit_state_normalization(
        np.zeros(6, np.float32), np.ones(6, np.float32), training_episode_ids=("a",)
    )

    class Arrays:
        states = np.zeros((4, 6), np.float32)
        mask = np.ones((4, 6), bool)
        frames = {"onboard_rgb": np.zeros((4, 2, 2, 3), np.uint8)}

    before = source.calls
    cloning.evaluate_policy(p, Arrays(), np.arange(4), None, np.zeros((4, 7), np.float32))
    assert source.calls > before, "a trainable encoder must be re-evaluated, not cached"
    # NB-A: torch.enable_grad() overrides an enclosing no_grad, so validation must ask for
    # inference explicitly or it retains activations for a tensor detached one line later.
    assert source.last_inference is True, "validation must not build an autograd graph"


def test_the_control_loop_does_not_build_an_autograd_graph():
    """predict_free is the one method whose entire contract is "this is inference".

    A trainable source's ``enable_grad()`` overrides the ``no_grad`` inside ``predict_free``,
    so without the explicit flag an A3 policy builds a graph on every step of a 50 ms loop.
    """
    source = MovingEncoder()
    p = ClonedPolicy(schema(), source, device="cpu", seed=0)
    p.fit_state_normalization(
        np.zeros(6, np.float32), np.ones(6, np.float32), training_episode_ids=("a",)
    )
    state = RobotState(
        np.zeros((1, 6), np.float32), np.ones((1, 6), bool), np.array([1.0]), p.state_schema
    )
    p.predict_free({"onboard_rgb": np.zeros((1, 2, 2, 3), np.uint8)}, state)
    assert source.last_inference is True, "the control loop must request inference"


# ---------------------------------------------------------------------------------------
# Behavioural tests for scripts/evaluate_policy.py, replacing the two source-text tripwires
# now that the runner exists -- the debt recorded against them in
# benchmarks/manifests/apple-policy-v1.json, come due.
#
# They live HERE and not in tests/test_policy_preflight.py because importing the runner pulls
# in embodied_jepa.policy and therefore torch, and that file is deliberately torch-free so it
# runs in the core CI job. This file already guards with importorskip("torch").
#
# One of the two cannot be written in its intended form: the "resets cohort C from stored
# values" check has no cohort-C code path to exercise, because the runner REFUSES cohort C.
# What is asserted instead is that refusal, and the stored-values behavioural test is STILL
# OWED when cohort-C support is built.
# ---------------------------------------------------------------------------------------
EVALUATE_POLICY = ROOT / "scripts" / "evaluate_policy.py"


def runner():
    """Import the runner by path; it is a script, not a package module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_evaluate_policy", EVALUATE_POLICY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_clips_to_the_protocols_bounds_not_the_contracts():
    """BEHAVIOURAL, replacing the text tripwire.

    ``ClonedPolicy.act`` clips to the contract's [-1, 1]; the protocol's frozen right-arm
    bounds are +-0.5. A runner executing a policy's raw output would command twice the delta
    every prior generation operated under, and the action would still be contract-valid, so
    nothing downstream would object.
    """
    module = runner()
    lower = np.array([0.0] * 6 + [-0.5] * 6 + [-1.0, -1.0], np.float32)
    upper = np.array([0.0] * 6 + [0.5] * 6 + [-1.0, 1.0], np.float32)

    saturated = np.zeros(14, np.float32)
    saturated[6:12] = 1.0  # what a saturated head emits after act()'s contract clip
    saturated[12] = -1.0
    saturated[13] = 1.0
    clipped, was_clipped = module.clip_to_configured_bounds(saturated, lower, upper)

    assert was_clipped is True
    assert np.all(clipped[6:12] == 0.5), "right-arm deltas must be clipped to the bound"
    assert clipped[13] == 1.0, "the grasp bound is +-1 and must not be narrowed"
    assert np.all(clipped[:6] == 0.0) and clipped[12] == -1.0

    # A command already inside the bounds is passed through untouched and not counted.
    inside = np.zeros(14, np.float32)
    inside[6:12] = 0.25
    inside[12] = -1.0
    passed, touched = module.clip_to_configured_bounds(inside, lower, upper)
    assert touched is False and np.array_equal(passed, inside)


def test_the_runner_reconstructs_every_arm_shape_correctly():
    """Enumerates ALL FOUR arm shapes, which is what the original smoke did not do.

    The runner's frozen-flag derivation was inverted, so A1, A2 and A3 -- three of four,
    including the primary arm -- could not load. It went unnoticed because the smoke ran A0,
    the only arm that still worked, and "it ran" and "it ran for each case" look identical in
    a terminal.

    The lesson generalises: **a smoke over a set of configured variants must enumerate the
    variants and assert it covered them.** This test is that assertion, in a form that needs
    neither a checkpoint nor a simulator.
    """
    module = runner()
    # (arm, encoder mode, whether cloning.train saves encoder weights, expected frozen flag)
    shapes = [
        ("A0_proprio_only", "none", False, None),
        ("A1_random_encoder", "random", False, True),
        ("A2_bc_frozen_e0", "frozen", False, True),
        ("A3_bc_finetuned_e0", "finetune", True, False),
    ]
    covered = set()
    for arm, encoder, saves_weights, expected in shapes:
        # This is exactly what cloning.train decides, restated as the invariant under test.
        trains_encoder = encoder == "finetune"
        assert saves_weights is trains_encoder
        if expected is None:
            covered.add(arm)
            continue
        checkpoint = {"feature_source_weights": {"w": 1} if saves_weights else None}
        assert module.frozen_flag_for(checkpoint) is expected, (
            f"{arm}: runner would build frozen={module.frozen_flag_for(checkpoint)} but the "
            f"checkpoint was trained frozen={expected}; ClonedPolicy.load rejects the mismatch"
        )
        covered.add(arm)
    assert covered == {a for a, _, _, _ in shapes}, "the enumeration must cover every arm"


def test_the_live_loop_executes_the_CLIPPED_command_not_the_raw_one():
    """The debt the manifest calls blocking, discharged at the level it specifies.

    The manifest requires asserting "what ACTUALLY REACHES THE EMBODIMENT is clipped to the
    bound". A test of ``clip_to_configured_bounds`` alone does not do that: an independent
    probe changed ``run_attempt`` to project and execute the RAW command while still counting
    the clip, and the entire suite stayed green at 782 passed while the robot received twice
    the configured delta and the report's clip counter actively lied.

    Bounds are read from the frozen config rather than restated here, so a widened config
    cannot pass silently.
    """
    import yaml

    module = runner()
    frozen = yaml.safe_load((ROOT / "configs" / "apple_wm_v4.yaml").read_text())["planner"]
    lower = np.asarray(frozen["lower_bounds"], np.float32)
    upper = np.asarray(frozen["upper_bounds"], np.float32)
    assert float(upper[6]) == 0.5, "the frozen right-arm bound is the thing under test"

    executed = []

    class Robot:
        def observe(self):
            return SimpleNamespace(images={}, state=None)

        def project_candidates(self, candidates):
            # The real projection only ever scales arm deltas toward zero, so passing the
            # candidate through is the permissive case: if the loop is wrong, it shows here.
            return SimpleNamespace(
                feasible=np.ones((1, 1), bool), actions=np.asarray(candidates, np.float32)
            )

        def execute(self, action):
            executed.append(np.asarray(action, np.float32).copy())
            return SimpleNamespace(applied_action=action, status="ok", reason=None)

        def stop(self, reason):
            pass

    class Saturated:
        """What a saturated head emits after act()'s contract clip to [-1, 1]."""

        def act(self, images, state):
            action = np.zeros(14, np.float32)
            action[6:12] = 1.0
            action[12] = -1.0
            action[13] = 1.0
            return action

    class Scorer:
        def evaluate(self):
            return {"success": False, "grasp": False}

    result = module.run_attempt(
        Saturated(),
        Robot(),
        Scorer(),
        lower=lower,
        upper=upper,
        max_steps=3,
        deadline_seconds=99.0,
    )

    assert len(executed) == 3
    for command in executed:
        assert float(np.abs(command[6:12]).max()) == 0.5, (
            "the embodiment received an unclipped right-arm delta: the loop is executing the "
            "raw command, not the clipped one"
        )
        assert float(command[13]) == 1.0, "the grasp bound is +-1 and must not be narrowed"
    assert result["clipped_commands"] == 3


def test_the_runner_rejects_a_command_that_moves_a_pinned_component():
    """The pinned components are frozen by the bounds AND by the policy module."""
    module = runner()
    lower = np.array([0.0] * 6 + [-0.5] * 6 + [-1.0, -1.0], np.float32)
    upper = np.array([0.0] * 6 + [0.5] * 6 + [-1.0, 1.0], np.float32)
    rogue = np.zeros(14, np.float32)
    rogue[0] = 0.3  # a left-arm delta, which nothing in this protocol may command
    with pytest.raises(ContractError, match="pinned component"):
        module.clip_to_configured_bounds(rogue, lower, upper)


def test_the_frozen_cohort_refusal_holds_at_the_type_boundary():
    """The most dangerous defect found in this task, and its class.

    The refusal originally tested membership BEFORE coercion, so ``"45300"`` was not equal to
    ``45300``, the intersection was empty, and the coercion that followed produced the frozen
    reset. Every other defect in this protocol costs a day; that one **silently consumes the
    frozen cohort**, which is unrecoverable.

    The class, which is the part worth keeping: **a guard that enumerates what is FORBIDDEN
    fails open; a guard that enumerates what is PERMITTED fails closed.** The blacklist would
    equally have admitted 45400 or 46000 -- anything nobody thought to forbid. The whitelist is
    right not because it happens to catch the string case but because refusal is its default.
    """
    module = runner()
    for form in (45300, "45300", 45300.0, np.int64(45300), np.int32(45339)):
        with pytest.raises(ContractError, match="FROZEN gating cohort"):
            module.cohort_resets((form,))

    # A permitted seed in any non-canonical form is still permitted -- the guard must not be
    # merely strict, it must be correct.
    # Whitespace coerces to a valid development seed, so it must be ACCEPTED -- a guard that
    # refuses legitimate input is a different defect, not extra safety.
    for form in (45000, "45000", " 45000 ", 45000.0, np.int64(45000)):
        assert module.cohort_resets((form,))[45000]["object_xy"]

    # Fails CLOSED on anything unenumerated, which is the property the class note describes.
    for unknown in (45400, 46000, 0, -1, 99999, True):
        with pytest.raises(ContractError, match="not in the development cohort"):
            module.cohort_resets((unknown,))


def test_the_runner_refuses_the_frozen_cohort():
    """BEHAVIOURAL, and the strongest guarantee available at this scope.

    The stored-values test cannot be written yet -- there is no cohort-C code path to
    exercise, because this runner refuses cohort C outright. That refusal is what is asserted
    here. **The stored-values behavioural test is still owed** when cohort-C support is built;
    see benchmarks/manifests/apple-policy-v1.json.
    """
    module = runner()
    assert set(module.COHORT_C) == set(range(45300, 45340))
    assert not set(module.COHORT_D) & set(module.COHORT_C)

    with pytest.raises(ContractError, match="FROZEN gating cohort"):
        module.cohort_resets((45300,))
    with pytest.raises(ContractError, match="FROZEN gating cohort"):
        module.cohort_resets((45000, 45339))  # one development seed does not excuse the other

    development = module.cohort_resets(module.COHORT_D)
    assert len(development) == 16


def test_the_runner_imports_the_committed_reset_rule_rather_than_reimplementing_it():
    """The one tripwire hole a parser cannot close.

    A reimplementation of ``default_rng(seed).uniform(...)`` references nothing and would
    silently produce a different cohort. This asserts the runner's resets equal the committed
    generator's, and that the runner contains no reset arithmetic of its own.
    """
    module = runner()
    evaluator = module.load_wide_reset()
    for seed in (45000, 45107):
        assert module.cohort_resets((seed,))[seed] == evaluator(seed)

    # PARSED, not grepped -- and this test caught itself making that mistake. The first
    # version checked for the TEXT "default_rng", which fired on the runner's own docstring
    # explaining what it must not do: the exact false positive that was removed from the
    # tripwire one commit earlier, reintroduced in the test replacing it.
    called = set()
    for node in ast.walk(ast.parse(source := EVALUATE_POLICY.read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            called.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", ""))
    assert "default_rng" not in called, (
        "scripts/evaluate_policy.py calls default_rng: the reset rule must be IMPORTED from "
        "the committed generator, never reimplemented. A reimplementation references nothing "
        "and no parser can catch it."
    )
    assert "wide_reset" in source, "the committed generator must be reached by name"
