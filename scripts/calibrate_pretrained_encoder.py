"""TASK-063 pre-freeze calibration: label-free facts about the pinned DINOv2 ViT-S/14 and its
seed-0 random-init floor. Nothing is fitted, and no target or label is read.

What it does:

* checks the 190 train + val roots of the frozen plan against the dataset's splits;
* renders, through TASK-062's calibration renderer (TASK-061's ``make_robot``, the renderer
  warm-up -- every renderer renders once and discards the result -- and the 8-command look),
  each root's post-look 112 px onboard frame and its apple-hidden and plate-hidden twins. Each
  root's reset coordinates are used only to render. The post-look frames must hash to TASK-062's
  calibration value;
* loads the pinned pretrained encoder (every file's sha256 checked) and builds the seed-0
  random init; records both weight digests;
* reports, per representation, TASK-062's label-free statistics (``S_apple``, ``S_plate``,
  ``R_apple``, ``R_plate``, effective rank over the 190 post-look frames): raw pixels, and for
  both encoders the patch-embedding tokens, the tokens after every block, the final tokens, their
  mean and the CLS token;
* times the CPU forward pass (for the budget), and checks that a second pass is bit-identical;
* re-checks, if the original FAIR checkpoint has been fetched, that the converted weights equal
  it tensor for tensor (``scripts/fetch_dinov2.py --verify-official``).

No readout is fitted, and no apple position, expert command or label sidecar enters any
statistic. The test split is not decoded; cohorts C and D are not simulated.

    uv run --no-sync python scripts/calibrate_pretrained_encoder.py \
        --output outputs/task063-pretrained-encoder/calibration-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import pretrained_encoder as pe  # noqa: E402

STORE = ROOT / "data" / "apple-wide-v1"
TASK062_CALIBRATION = ROOT / "outputs" / "task062-encoder-study" / "calibration-v2.json"


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")
CAL = _load("_calibrate_encoder_study", "scripts/calibrate_encoder_study.py")
FETCH = _load("_fetch_dinov2", "scripts/fetch_dinov2.py")


def timed_features(model, frames) -> tuple[dict, float]:
    begin = time.perf_counter()
    out = pe.features(model, frames, hidden_states=True)
    return out, time.perf_counter() - begin


def main() -> int:
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    begin = time.time()
    roots, _collector = BASE.plan_roots()
    dataset = json.loads((STORE / "meta" / "jepa_manifest.json").read_text())
    ic.check_split(roots, dataset["splits"])
    frames, plates = CAL.render_post_look(roots)
    post_sha = hashlib.sha256(frames["post"].tobytes()).hexdigest()
    task062 = json.loads(TASK062_CALIBRATION.read_text())
    result = {
        "task": "TASK-063",
        "what": "label-free encoder facts; no target read, no readout fitted",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "roots": len(roots),
        "plate_bodies_hidden": plates,
        "ablation_renderer_reproduces_post_look": int(sum(frames["reproduces"])),
        "post_look_frames_sha256": post_sha,
        "post_look_frames_equal_task062_calibration": post_sha
        == task062["post_look_frames_sha256"],
        "weights": {
            "repository": pe.HF_REPOSITORY,
            "revision": pe.HF_REVISION,
            "files_sha256": pe.check_files(pe.DEFAULT_DIRECTORY),
        },
        "preprocessing": {
            "frame": "uint8 [N,112,112,3] RGB post-look onboard frame",
            "input_size": pe.INPUT_SIZE,
            "resize": "torch bicubic, align_corners=False, antialias=False, no crop",
            "normalisation": {"mean": pe.IMAGENET_MEAN, "std": pe.IMAGENET_STD},
            "token_grid": [pe.GRID, pe.GRID],
            "width": pe.WIDTH,
        },
        "representations": {},
        "timing_seconds_cpu": {},
    }
    official = pe.DEFAULT_DIRECTORY / FETCH.OFFICIAL_FILE
    if official.is_file():
        result["weights"]["official_checkpoint"] = {
            "url": FETCH.OFFICIAL_URL,
            "sha256": pe.sha256(official),
            "pinned_sha256": FETCH.OFFICIAL_SHA256,
            "equivalence": FETCH.compare_with_official(
                pe.DEFAULT_DIRECTORY / "model.safetensors", official
            ),
        }
    flat = lambda a: a.reshape(len(a), -1).astype(np.float64) / 255.0  # noqa: E731
    result["representations"]["raw_pixels"] = CAL.shares(
        flat(frames["post"]), flat(frames["apple_hidden"]), flat(frames["plate_hidden"])
    )
    encoders = {"pretrained": pe.load_pretrained(), "random_init": pe.random_init()}
    for name, model in encoders.items():
        stages, seconds = {}, {}
        for key in ("post", "apple_hidden", "plate_hidden"):
            stages[key], seconds[key] = timed_features(model, frames[key])
        again = pe.features(model, frames["post"])
        result["representations"][name] = {
            stage: CAL.shares(
                stages["post"][stage], stages["apple_hidden"][stage], stages["plate_hidden"][stage]
            )
            for stage in stages["post"]
        }
        result["representations"][name]["_weights_digest"] = pe.weights_digest(model)
        result["representations"][name]["_second_pass_bit_identical"] = all(
            np.array_equal(again[k], stages["post"][k]) for k in pe.READOUT_POINTS
        )
        result["timing_seconds_cpu"][name] = seconds
    result["elapsed_seconds"] = time.time() - begin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        name: {
            stage: {k: round(v[k], 3) for k in ("S_apple", "S_plate", "effective_rank")}
            for stage, v in reps.items()
            if not stage.startswith("_") and stage in ("cls", "tokens", "tokens_mean")
        }
        for name, reps in result["representations"].items()
        if name != "raw_pixels"
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
