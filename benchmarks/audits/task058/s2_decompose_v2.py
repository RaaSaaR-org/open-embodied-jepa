"""Recompute the TASK-050 encoder/rollout decomposition (never written to any artifact) with the
3b6af0b code, plus model-own persistence baselines for G3/G4/G5/G6. Inference only, val only."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent / "v2tree"
os.chdir(HERE)
sys.path.insert(0, str(HERE / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from embodied_jepa import world_model_v2 as w  # noqa: E402
from embodied_jepa.config import MODELS  # noqa: E402

def main():
    CONFIGS = {
        "lewm_onboard": ("configs/apple_wm_v2_lewm.yaml", "checkpoints/task050-wm-v2/leworldmodel.pt"),
        "lewm_hand_crop": (
            "configs/apple_wm_v2_lewm_hand_crop.yaml",
            "checkpoints/task050-wm-v2/leworldmodel_hand_crop.pt",
        ),
        "native_onboard": ("configs/apple_wm_v2.yaml", "checkpoints/task050-wm-v2/native_jepa.pt"),
    }
    arm = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "mps"
    cfg, ckpt = CONFIGS[arm]
    config, store, settings, camera = w._open(cfg)
    env = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = MODELS.create(config.backend, state_schema=store.state_schema, device=device,
                          seed=env["seed"], config=settings, metadata=env["metadata"])
    model.load(ckpt)
    arrays = w.load_split(store, "val", camera, workers=8, acknowledge_privileged_training_labels=True)
    windows = w.window_starts(arrays, max(w.HORIZONS), stride=4)
    H = 8
    starts = arrays.offsets[windows[:, 0]] + windows[:, 1]
    tg = starts + H
    T = arrays.targets
    dropped = T["apple_dropped"][:, 0] > 0.5
    valid = ~dropped[starts] & ~dropped[tg]
    disp = np.linalg.norm(T["palm_minus_apple"][tg] - T["palm_minus_apple"][starts], axis=1)
    moving = valid & (disp >= w.MOVING_THRESHOLD_M)
    phase = arrays.phase[tg]
    grasp = valid & np.isin(phase, (w.PHASE_CLOSE, w.PHASE_LIFT))
    lift = valid & (phase == w.PHASE_LIFT)
    approach = valid & np.isin(phase, (w.PHASE_ORIENT, w.PHASE_DESCEND))
    actions = w._window_actions(arrays, windows, max(w.HORIZONS))
    pred = w.predicted_readouts(model, arrays, starts, actions, camera, store.state_schema)
    es = w.readout_rows(model, arrays, starts, camera, store.state_schema)
    et = w.readout_rows(model, arrays, tg, camera, store.state_schema)
    med = w._median


    def err(a, b):
        return np.linalg.norm(a - b, axis=1)


    pma_t = T["palm_minus_apple"][tg]
    pma_s = T["palm_minus_apple"][starts]
    rollout = err(pred["palm_minus_apple"][:, H - 1], pma_t)
    r = {
        "arm": arm, "device": device, "valid": int(valid.sum()), "moving": int(moving.sum()),
        "G1_rollout_moving": med(rollout[moving]),
        "persistence_moving": med(err(es["palm_minus_apple"], pma_t)[moving]),
        "encoded_start_vs_start_truth_moving": med(err(es["palm_minus_apple"], pma_s)[moving]),
        "encoded_target_vs_target_truth_moving": med(err(et["palm_minus_apple"], pma_t)[moving]),
        "rollout_all_valid": med(rollout[valid]),
        "encoded_target_all_valid": med(err(et["palm_minus_apple"], pma_t)[valid]),
        "persistence_all_valid": med(err(es["palm_minus_apple"], pma_t)[valid]),
        "rollout_nonmoving": med(rollout[valid & ~moving]),
        "encoded_target_nonmoving": med(err(et["palm_minus_apple"], pma_t)[valid & ~moving]),
        "frac_windows_rollout_le_encoded_target_moving": float(
            (rollout <= err(et["palm_minus_apple"], pma_t))[moving].mean()),
        "G3_rollout": med(err(pred["apple_minus_plate"][:, H - 1], T["apple_minus_plate"][tg])[valid]),
        "G3_model_persistence": med(err(es["apple_minus_plate"], T["apple_minus_plate"][tg])[valid]),
        "G3_encoded_target": med(err(et["apple_minus_plate"], T["apple_minus_plate"][tg])[valid]),
        "G4_rollout": med(err(pred["apple_height"][:, H - 1], T["apple_height"][tg])[grasp]),
        "G4_model_persistence": med(err(es["apple_height"], T["apple_height"][tg])[grasp]),
    }
    held = T["apple_held"][tg][:, 0] > 0.5
    r["G5_rollout"] = w.auroc(pred["apple_held"][:, H - 1, 0][lift], held[lift])
    r["G5_model_persistence"] = w.auroc(es["apple_held"][:, 0][lift], held[lift])
    r["G5_encoded_target"] = w.auroc(et["apple_held"][:, 0][lift], held[lift])
    off = w.APPROACH_OFFSET_M
    ct = np.linalg.norm(pma_t - off, axis=1)
    rows = approach & (ct >= w.COST_FLOOR_M)
    for name, v in (("rollout", pred["palm_minus_apple"][:, H - 1]), ("model_persistence", es["palm_minus_apple"]),
                    ("encoded_target", et["palm_minus_apple"])):
        cp = np.linalg.norm(v - off, axis=1)
        r[f"G6_{name}"] = med(np.abs(np.log(np.maximum(cp, 1e-4) / np.maximum(ct, 1e-4)))[rows])
        r[f"G6_{name}_median_pred_cost_cm"] = 100 * med(cp[rows])
    r["G6_rows"] = int(rows.sum())
    r["G6_true_cost_median_cm"] = 100 * med(ct[rows])
    out = Path(__file__).parent / f"s2_decompose_{arm}_{device}.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
