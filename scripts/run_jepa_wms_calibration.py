"""TASK-084 (Phase 0): Meta's released JEPA-WM on Push-T, with its planner and with ours.

Runs the upstream JEPA-WMs planning evaluation (facebookresearch/jepa-wms, CC BY-NC 4.0, used
only at run time from an ignored checkout; nothing of it is copied here) for one arm:

- ``upstream``: the upstream ``CEMPlanner``, unchanged.
- ``ours``: ``embodied_jepa.planning.CEMPlanner`` behind a thin bridge that gives it the upstream
  planner interface. Only the optimiser changes; the model, the episodes, the cost (the upstream
  L2 objective with alpha = 0.1, read at the last predicted step), the action execution and the
  success check are upstream's.

The protocol is ``docs/experiments/jepa_wms_pusht_calibration.md``. This script must run in the
separate Python 3.10 runtime that holds the upstream dependency set (see that document), with this
checkout's ``src`` on ``PYTHONPATH``; it is not part of the core package and the core import check
does not see it.

Per-episode rows go to ``<output>/episodes.jsonl`` and a summary to ``<output>/summary.json``.
The script refuses to write into an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE_CONFIG = (
    "configs/evals/simu_env_planning/pt/jepa-wm/"
    "pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
)
UPSTREAM_REVISION = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
CHECKPOINT_SHA256 = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
# The frozen DINOv2 ViT-S/14 the model encodes with: weights (the repository's pinned file) and the
# torch-hub code tree upstream loads them with (sha256 of the sorted per-file sha256 list of *.py).
DINOV2_WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINOV2_HUB_CODE_SHA256 = "5c0d48cae201d8c81d185589ecad6fd85d3cee63365a20b71ed7ce3f863abbc9"
# The Push-T data: the DINO-WM release on OSF (sha256 as OSF publishes it), unzipped.
DATASET_ZIP_SHA256 = "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08"
# The bridge's [-1, 1] -> normalised-action scale: the 99.5th percentile of |z| per raw action
# dimension over the train split's relative actions, z-scored with upstream's fixed constants.
SCALE_QUANTILE = 0.995
OURS_SEED_OFFSET = 84_000_000
UNROLL_CHUNK = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hub_code_sha256(folder: Path) -> str:
    lines = "".join(
        f"{sha256(p)}  ./{p.relative_to(folder).as_posix()}\n"
        for p in sorted(folder.rglob("*.py"), key=lambda q: q.relative_to(folder).as_posix())
    )
    return hashlib.sha256(lines.encode()).hexdigest()


def action_scale(dataset_root: Path) -> np.ndarray:
    """Per raw action dimension bound b_d in upstream's normalised action units (train split)."""
    import inspect
    import pickle

    import torch
    from app.plan_common.datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PushTDataset

    train = dataset_root / "pusht_noise" / "train"
    # As upstream's PushTDataset does: divide by the env's action scale before z-scoring.
    env_action_scale = inspect.signature(PushTDataset).parameters["action_scale"].default
    actions = torch.load(train / "rel_actions.pth", map_location="cpu").float() / env_action_scale
    with (train / "seq_lengths.pkl").open("rb") as handle:
        lengths = pickle.load(handle)
    rows = torch.cat([actions[i, : int(n)] for i, n in enumerate(lengths)], dim=0)
    z = (rows - ACTION_MEAN) / ACTION_STD
    return np.quantile(z.abs().numpy(), SCALE_QUANTILE, axis=0).astype(np.float64)


def make_bridge(scale_raw: np.ndarray, seed_base: int):
    """Build the upstream-interface planner class that runs this repository's CEMPlanner."""
    import torch
    from evals.simu_env_planning.planning.planning.planner import Planner, PlanningResult

    from embodied_jepa.contracts import ACTION_DIM
    from embodied_jepa.planning import CEMConfig, CEMPlanner

    class _LatentModel:
        """Gives CEMPlanner the model contract it calls: predict() and distance()."""

        def __init__(self, bridge):
            self.bridge = bridge

        def predict(self, latent, candidates):
            bridge = self.bridge
            batch, samples, horizon, _ = candidates.shape
            if batch != 1:
                raise ValueError("the bridge plans one episode at a time")
            used = candidates[0, :, :, : bridge.action_dim] * bridge.scale
            actions = torch.as_tensor(np.ascontiguousarray(used.transpose(1, 0, 2)))
            actions = actions.to(device="cuda", dtype=torch.float32)  # [H, S, A]
            return bridge.unroll(latent, actions), horizon

        def distance(self, predictions, goal_latent):
            encodings, horizon = predictions
            cost = self.bridge.objective(encodings, None, keepdims=True)  # [tau + H, S]
            cost = cost[-horizon:].transpose(0, 1).unsqueeze(0)  # [1, S, H]
            return cost.float().cpu().numpy()

    class OursCEMBridge(Planner):
        def __init__(
            self,
            unroll,
            action_dim,
            iterations,
            num_samples,
            num_elites,
            horizon,
            num_act_stepped,
            **_ignored,
        ):
            super().__init__(unroll)
            if action_dim > ACTION_DIM:
                raise ValueError(f"model action_dim {action_dim} exceeds {ACTION_DIM}")
            if action_dim % scale_raw.shape[0]:
                raise ValueError("model action_dim is not a multiple of the raw action dim")
            self.action_dim = action_dim
            self.scale = np.tile(scale_raw, action_dim // scale_raw.shape[0]).astype(np.float32)
            self.iterations, self.num_samples = int(iterations), int(num_samples)
            self.num_elites, self.horizon = int(num_elites), int(horizon)
            self.num_act_stepped = int(num_act_stepped)
            self.seed_base, self.calls = int(seed_base), 0
            self.decode_each_iteration = False

        @torch.no_grad()
        def plan(self, z_init, steps_left=None):
            horizon = self.horizon if steps_left is None else min(self.horizon, steps_left)
            config = CEMConfig(
                horizon=horizon,
                samples=self.num_samples,
                iterations=self.iterations,
                elites=self.num_elites,
                seed=self.seed_base + self.calls,
            )
            self.calls += 1
            plan = CEMPlanner(config).plan(_LatentModel(self), z_init, None)
            chosen = plan.actions[0, :, : self.action_dim] * self.scale
            actions = torch.as_tensor(chosen, dtype=torch.float32, device="cuda")
            return PlanningResult(
                actions=actions[: self.num_act_stepped],
                losses=torch.tensor([[float(plan.costs[0])]]),
                prev_elite_losses_mean=torch.zeros(1, 1),
                prev_elite_losses_std=torch.zeros(1, 1),
                pred_frames_over_iterations=None,
                predicted_best_encs_over_iterations=[],
            )

    return OursCEMBridge


def build_config(runtime: Path, output: Path, checkpoint: Path, seed: int, episodes: int) -> dict:
    import yaml

    with (runtime / "jepa-wms" / BASE_CONFIG).open() as handle:
        cfg = yaml.safe_load(handle)
    cfg["folder"] = str(output / "upstream_logs")
    cfg["tag"] = "task084"
    cfg["meta"]["seed"] = int(seed)
    cfg["meta"]["eval_episodes"] = int(episodes)
    cfg["logging"]["optional_plots"] = False
    cfg["model_kwargs"]["checkpoint"] = str(checkpoint)
    # Visualisation only: the decoder heads are not released for Push-T (state head) or are a
    # 3.6 GB download (image head), and planning never reads them.
    cfg["model_kwargs"]["pretrain_kwargs"]["heads_cfg"] = {}
    cfg["planner"]["decode_each_iteration"] = False
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--arm", choices=("upstream", "ours"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime, output = args.runtime.resolve(), args.output.resolve()
    if output.exists():
        print(f"refusing: {output} exists", file=sys.stderr)
        return 2
    upstream = runtime / "jepa-wms"
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        print(f"refusing: upstream checkout is at {revision}", file=sys.stderr)
        return 2
    checkpoint = runtime / "downloads" / "jepa_wm_pusht.pth.tar"
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA256:
        print("refusing: checkpoint hash mismatch", file=sys.stderr)
        return 2
    hub = runtime / "hub" / "hub"
    pins = {
        "dinov2_weights": (
            sha256(hub / "checkpoints" / "dinov2_vits14_pretrain.pth"),
            DINOV2_WEIGHTS_SHA256,
        ),
        "dinov2_hub_code": (
            hub_code_sha256(hub / "facebookresearch_dinov2_main"),
            DINOV2_HUB_CODE_SHA256,
        ),
        "dataset_zip": (sha256(runtime / "downloads" / "pusht_noise.zip"), DATASET_ZIP_SHA256),
    }
    for name, (got, want) in pins.items():
        if got != want:
            print(f"refusing: {name} sha256 {got} != {want}", file=sys.stderr)
            return 2
    output.mkdir(parents=True)

    os.environ.setdefault("JEPAWM_DSET", str(runtime / "dset"))
    os.environ.setdefault("JEPAWM_LOGS", str(runtime / "logs"))
    os.environ.setdefault("JEPAWM_HOME", str(runtime))
    os.environ.setdefault("TORCH_HOME", str(runtime / "hub"))
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)

    import evals.simu_env_planning.planning.gc_agent as gc_agent
    import torch
    from evals.simu_env_planning.eval import main as eval_main
    from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
    from src.utils.distributed import init_distributed

    os.environ.setdefault("MASTER_PORT", "29584")
    init_distributed(port=int(os.environ["MASTER_PORT"]), rank_and_world_size=(0, 1))

    scale_raw = action_scale(Path(os.environ["JEPAWM_DSET"]))
    if args.arm == "ours":
        gc_agent.CEMPlanner = make_bridge(scale_raw, OURS_SEED_OFFSET + 1000 * args.seed)

    rows: list[dict] = []
    stash: dict = {}
    original_sample = PlanEvaluator.sample_traj_segment_from_dset
    original_eval = PlanEvaluator.eval

    def sample(self, *a, **k):
        obs, state, act, info = original_sample(self, *a, **k)
        stash["init_state"] = np.asarray(state[0], dtype=np.float64).tolist()
        stash["goal_state"] = np.asarray(state[-1], dtype=np.float64).tolist()
        return obs, state, act, info

    def evaluate(self, cfg, agent, env, task_idx=-1, ep=0):
        stash.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_eval(self, cfg, agent, env, task_idx=task_idx, ep=ep)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        row = {
            "episode": int(ep),
            "ep_seed": int(ep_seed),
            "success": bool(result[1]),
            "state_dist": float(result[8]),
            "seconds": elapsed,
            "init_state": stash.get("init_state"),
            "goal_state": stash.get("goal_state"),
        }
        rows.append(row)
        with (output / "episodes.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return result

    # The upstream evaluation ran on 80 GB GPUs; one 300-sample unroll does not fit in 16 GB.
    # Split the sample axis into chunks: every sample's rollout is independent of the others, so
    # this changes memory use only (and floating-point reduction order, negligibly).
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
    from tensordict import TensorDict

    original_unroll = EncPredWM.unroll

    def chunked_unroll(self, z_ctxt, act_suffix=None, debug=False):
        if act_suffix is None or act_suffix.shape[1] <= UNROLL_CHUNK:
            return original_unroll(self, z_ctxt, act_suffix=act_suffix, debug=debug)
        parts = [
            original_unroll(self, z_ctxt, act_suffix=chunk, debug=debug)
            for chunk in torch.split(act_suffix, UNROLL_CHUNK, dim=1)
        ]
        if isinstance(parts[0], torch.Tensor):
            return torch.cat(parts, dim=1)
        return TensorDict(
            {key: torch.cat([part[key] for part in parts], dim=1) for key in parts[0].keys()}
        )

    EncPredWM.unroll = chunked_unroll
    PlanEvaluator.sample_traj_segment_from_dset = sample
    PlanEvaluator.eval = evaluate

    cfg = build_config(runtime, output, checkpoint, args.seed, args.episodes)
    started = time.time()
    eval_main(cfg)
    wall = time.time() - started

    ours_rev = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    successes = sum(r["success"] for r in rows)
    summary = {
        "task": "TASK-084",
        "arm": args.arm,
        "meta_seed": args.seed,
        "episodes_requested": args.episodes,
        "episodes_run": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "wall_seconds": wall,
        "action_scale_raw": scale_raw.tolist(),
        "scale_quantile": SCALE_QUANTILE,
        "planner": cfg["planner"],
        "upstream_revision": revision,
        "checkpoint_sha256": checkpoint_sha,
        "pins_sha256": {name: got for name, (got, _) in pins.items()},
        "repository_revision": ours_rev,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "config": cfg,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("arm", "meta_seed", "episodes_run", "successes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
