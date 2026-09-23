"""Shared action-conditioned prediction-step options (TASK-054).

These are synthetic contract tests on a tiny native model. They establish what the
options *do* to the rollout and to the loss, not that any of them helps: nothing here
is a learned result.

The load-bearing property is the first one. Every option defaults to off, and with them
off the rollout and the multistep loss must be bit-for-bit what they were before
TASK-054, or every earlier checkpoint's numbers stop meaning what they said.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_jepa.contracts import ContractError, StateSchema  # noqa: E402
from embodied_jepa.models import NativeJEPA  # noqa: E402

SCHEMA = StateSchema(("arm.q",), ("rad",), "fixture_v0")
BASE = {"image_size": 32, "latent_dim": 16, "hidden_dim": 24, "max_horizon": 8}


def build(**overrides):
    return NativeJEPA(SCHEMA, seed=5, config=BASE | overrides)


def rollout_inputs(model, steps=6, count=4, seed=11):
    rng = np.random.default_rng(seed)
    initial = torch.from_numpy(rng.normal(size=(count, model.config["latent_dim"])).astype("f4"))
    actions = torch.from_numpy(rng.uniform(-1, 1, (count, steps, 14)).astype("f4"))
    return initial, actions


def test_every_option_is_off_by_default():
    model = build()
    assert model.config["action_chunk"] == 1
    assert model.config["predictor_step_embedding"] is False
    assert model.config["multistep_tail_weight"] == 0.0
    assert not hasattr(model, "action_chunk_encoder")
    assert not hasattr(model, "predictor_step")


def test_with_the_options_off_the_rollout_is_the_plain_autoregressive_loop():
    model = build()
    initial, actions = rollout_inputs(model)
    with torch.no_grad():
        state, expected = initial, []
        for step in range(actions.shape[1]):
            state = model.next_embedding(state, actions[:, step])
            expected.append(state)
        reference = torch.stack(expected, dim=1)
        assert torch.equal(model.rollout(initial, actions), reference)


def test_with_the_options_off_the_multistep_loss_is_the_plain_mean():
    model = build()
    predicted, targets = rollout_inputs(model)[0][:, None].expand(-1, 6, -1), None
    targets = torch.randn_like(predicted)
    assert model.multistep_weights(6) is None
    assert torch.equal(
        model.multistep_loss(predicted, targets), (predicted - targets).square().mean()
    )


def test_the_tail_weighting_keeps_the_total_weight_and_favours_late_steps():
    model = build(multistep_tail_weight=3.0)
    weights = model.multistep_weights(8)
    assert weights.shape == (1, 8, 1)
    flat = weights.reshape(-1)
    assert float(flat.mean()) == pytest.approx(1.0)
    assert float(flat[-1] / flat[0]) == pytest.approx(4.0)
    assert bool((flat[1:] > flat[:-1]).all())
    # A single predicted step has no tail to weight, so the option is inert there.
    assert model.multistep_weights(1) is None


def test_the_tail_weighting_moves_the_loss_towards_the_late_steps():
    plain, tailed = build(), build(multistep_tail_weight=3.0)
    predicted = torch.zeros(2, 4, plain.config["latent_dim"])
    targets = torch.zeros_like(predicted)
    late, early = targets.clone(), targets.clone()
    late[:, -1] = 1.0
    early[:, 0] = 1.0
    # The same total squared error, placed at the end rather than the beginning.
    assert plain.multistep_loss(predicted, late) == plain.multistep_loss(predicted, early)
    assert tailed.multistep_loss(predicted, late) > tailed.multistep_loss(predicted, early)


def test_the_action_chunk_conditions_a_step_on_its_whole_chunk_and_no_further():
    model = build(action_chunk=3)
    initial, actions = rollout_inputs(model, steps=6)
    with torch.no_grad():
        reference = model.rollout(initial, actions)
        # Action 2 lies inside step 0's chunk [0,3), so it must change step 0.
        inside = actions.clone()
        inside[:, 2] += 1.0
        assert not torch.equal(model.rollout(initial, inside)[:, 0], reference[:, 0])
        # Action 3 lies outside it, so step 0 must be untouched (later steps change).
        outside = actions.clone()
        outside[:, 3] += 1.0
        changed = model.rollout(initial, outside)
        assert torch.equal(changed[:, 0], reference[:, 0])
        assert not torch.equal(changed[:, 3], reference[:, 3])


def test_the_action_chunk_pads_past_the_end_of_the_sequence():
    model = build(action_chunk=4)
    initial, actions = rollout_inputs(model, steps=5)
    with torch.no_grad():
        context = model.action_chunk_context(actions)
        padded = torch.cat((actions, actions.new_zeros(actions.shape[0], 3, 14)), dim=1)
        for step in range(actions.shape[1]):
            chunk = padded[:, step : step + 4].reshape(actions.shape[0], -1)
            # allclose, not equal: the rollout batches the MLP over all steps at once,
            # which is a different matmul shape from this per-step reference.
            assert torch.allclose(context[:, step], model.action_chunk_encoder(chunk), atol=1e-6)


def test_the_step_embedding_starts_as_the_shared_single_step_rollout():
    plain, stepwise = build(), build(predictor_step_embedding=True)
    # Drawn last, so every earlier module is identical between the two arms.
    for name, parameter in plain.named_parameters():
        assert torch.equal(parameter, dict(stepwise.named_parameters())[name])
    initial, actions = rollout_inputs(plain)
    with torch.no_grad():
        assert torch.equal(stepwise.rollout(initial, actions), plain.rollout(initial, actions))
        stepwise.predictor_step.weight.normal_(generator=torch.Generator().manual_seed(2))
        assert not torch.equal(stepwise.rollout(initial, actions), plain.rollout(initial, actions))


def test_the_step_embedding_refuses_a_rollout_longer_than_it_was_built_for():
    model = build(predictor_step_embedding=True)
    initial, actions = rollout_inputs(model, steps=model.config["max_horizon"] + 1)
    with pytest.raises(ContractError, match="step embedding"):
        model.rollout(initial, actions)


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_chunk": 0},
        {"action_chunk": 1.5},
        {"action_chunk": 9},  # above max_horizon
        {"predictor_step_embedding": 1},
        {"multistep_tail_weight": -1.0},
        {"multistep_tail_weight": float("nan")},
    ],
)
def test_invalid_prediction_step_options_are_refused(overrides):
    with pytest.raises(ContractError):
        build(**overrides)


def test_the_options_round_trip_through_the_checkpoint_envelope(tmp_path):
    model = build(action_chunk=3, predictor_step_embedding=True, multistep_tail_weight=2.0)
    path = tmp_path / "model.pt"
    model.save(path)
    build(action_chunk=3, predictor_step_embedding=True, multistep_tail_weight=2.0).load(path)
    with pytest.raises(ContractError, match="config"):
        build(action_chunk=3, predictor_step_embedding=True).load(path)
