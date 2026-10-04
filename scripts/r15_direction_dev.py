"""Development measurements behind the DRAFT ruling R15 (no gate, no new simulation, CPU only).

Ruling document: ``docs/experiments/apple_lewm_next_v2_direction.md`` §3. Everything here is a
re-analysis of the C1 feasibility record's run-2 artifacts (``outputs/c1-run-2`` in the
``c1-feasibility`` worktree, checked against the sha256 recorded in
``docs/experiments/apple_lewm_next_v2_c1_feasibility.md`` §2.1) plus reset-value arithmetic on R15's
declared development seeds 58000-58511. No episode is simulated, no world model is trained or run,
and no GPU is used. Three measurements:

1. **The commit tolerance curve (development estimate).** Counted success against the landing miss
   |committed aim - plate(525)| over every completed single-commit attempt of run-2 (the ceiling,
   the proxies, their offset variants, the comparators and the 255 corpus roots), binned by
   magnitude. Checked against the measured mean- and shuf-proxy counts.
2. **The plate's spread at the decision step** under candidate conditions (v2's reset, TASK-074's
   -y move at 9 and 12 cm, a disc or -y half-disc move of radius rho), and the scene-blind twins'
   expected counts from the curve (their miss is p - p_bar or p - p', R15.2).
3. **The readout at r = 460** on run-2's corpus frames: cross-fitted ridge on DINOv2 tokens pooled
   to 2 x 2, 4 x 4, 8 x 8 and the full 16 x 16 grid (C1's folds, lambda grid and salts), the same
   readouts on the plate-hidden renders, a learning curve, and each readout's errors mapped
   through the curve.

Writes one JSON report and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RUN = Path("/home/huhn/develop/emai/worktrees/c1-feasibility/outputs/c1-run-2")
PINNED = {  # docs/experiments/apple_lewm_next_v2_c1_feasibility.md §2.1
    "report.json": "7779709cf62ee61fac3e2acfbe16b2049479d02f8dc0af47fd809ccb993fcef6",
    "corpus.npz": "93e96c3f82b476e2a5096d22d6eb09e04bfa0323ecfdd26b7a25885e3275cdb5",
}
SEEDS = (58000, 58511)  # R15.9: reset values only, nothing simulated
SALTS = {"learning_curve": 7801, "move_draw": 7802}  # R15.9
EDGES_CM = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, float("inf"))
RHO_CM = (3, 4, 5, 6)  # TASK-073's |d| grid (wm_critic_v2.SHIFT_GRID_CM), used as radii
GRIDS = (2, 4, 8, 16)
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def miss_vectors(report: dict, corpus) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(aim - plate(525)) in cm and the counted success of every completed single-commit attempt."""
    vec, ok, arm = [], [], []
    stages = report["stages"]
    groups = [("H-final", stages["ceiling"]["attempts"])]
    for stage in ("proxies", "comparators"):
        groups += [(k, v) for k, v in stages[stage]["attempts"].items() if k != "H-now-reaim"]
    for name, attempts in groups:
        for a in attempts:
            if a["kpred"].get("refused") or not a.get("commit"):
                continue
            vec.append(np.subtract(a["commit"]["target"], a["kpred"]["plate_at"]["525"]))
            ok.append(bool(a["success"]))
            arm.append(name)
    for t, p, s in zip(corpus["target"], corpus["plate_s1"], corpus["success"], strict=True):
        vec.append(t - p)
        ok.append(bool(s))
        arm.append("corpus")
    return 100.0 * np.array(vec), np.array(ok), arm


def tolerance_curve(miss_cm: np.ndarray, ok: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(EDGES_CM[:-1], EDGES_CM[1:], strict=True):
        k = (miss_cm >= lo) & (miss_cm < hi)
        n = int(k.sum())
        rows.append(
            {"lo_cm": lo, "hi_cm": hi, "n": n, "success": int(ok[k].sum()), "rate": ok[k].mean()}
            if n
            else {"lo_cm": lo, "hi_cm": hi, "n": 0, "success": 0, "rate": 0.0}
        )
    return rows


def expected_rate(curve: list[dict], miss_cm) -> np.ndarray:
    rates = np.array([r["rate"] for r in curve])
    i = np.searchsorted(np.array(EDGES_CM), np.asarray(miss_cm, float), side="right") - 1
    return rates[np.clip(i, 0, len(rates) - 1)]


def spreads(curve: list[dict]) -> dict:
    from embodied_jepa import lewm_planner_v2 as lp
    from embodied_jepa import wm_critic_v2 as wc

    seeds = range(SEEDS[0], SEEDS[1] + 1)
    base = np.array([wc.wide_reset_values(s)["plate_xy"] for s in seeds])
    plates = {"v2 reset (no move)": base}
    angles_by_cm = {}
    for cm in (9, 12):  # TASK-074's -y move: its own reset, direction rule and salt (7413)
        rows, angles = [], []
        for s in seeds:
            reset = lp.condition_reset(s)
            v = lp.shift_vector(s, reset, cm)
            rows.append(np.add(reset["plate_xy"], v))
            angles.append(float(np.degrees(np.arctan2(v[1], v[0])) % 360.0))
        plates[f"TASK-074 -y move, {cm} cm"] = np.array(rows)
        angles_by_cm[f"{cm} cm"] = np.percentile(angles, [5, 50, 95]).round(1).tolist()
    rng = np.random.default_rng(SALTS["move_draw"])
    for family, (lo, hi) in (("disc", (0.0, 2 * np.pi)), ("-y half-disc", (np.pi, 2 * np.pi))):
        for rho in RHO_CM:
            r = rho / 100.0 * np.sqrt(rng.uniform(size=len(base)))
            theta = rng.uniform(lo, hi, len(base))
            plates[f"{family}, rho {rho} cm"] = base + np.c_[r * np.cos(theta), r * np.sin(theta)]
    out = {}
    for name, p in plates.items():
        to_mean = 100.0 * np.linalg.norm(p - p.mean(axis=0), axis=1)
        to_next = 100.0 * np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
        out[name] = {
            "median_to_mean_cm": float(np.median(to_mean)),
            "p87_5_to_mean_cm": float(np.percentile(to_mean, 87.5)),
            "median_to_foreign_cm": float(np.median(to_next)),
            "mean_proxy_expected_of_32": float(32 * expected_rate(curve, to_mean).mean()),
            "shuf_proxy_expected_of_32": float(32 * expected_rate(curve, to_next).mean()),
        }
    out["TASK-074 direction percentiles 5/50/95 (deg)"] = angles_by_cm
    return out


def readouts(corpus, curve: list[dict]) -> dict:
    import torch

    from embodied_jepa import lewm_next_c1 as c1
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens
    from embodied_jepa.models.latent_critic import RidgeReadout
    from embodied_jepa.obs_ceiling_v2_offline import preprocess_view

    torch.set_num_threads(6)
    encoder = pe.load_pretrained().eval()

    def tokens(frames):
        out = []
        with torch.no_grad():
            for lo in range(0, len(frames), 16):
                last = encoder(pixel_values=preprocess_view(frames[lo : lo + 16]))
                out.append(last.last_hidden_state[:, 1:].reshape(-1, 256 * 384).float().numpy())
        return np.concatenate(out)

    visible, hidden = tokens(corpus["frames_r"]), tokens(corpus["hidden_r"])
    n = len(corpus["seeds"])
    roots = corpus["seeds"].astype(str)
    fold = c1.outer_folds(n)
    y = corpus["plate_r"]

    def ridge(x, yy, groups):
        return RidgeReadout.fit(
            x,
            yy,
            groups,
            lambdas=c1.LAMBDA_GRID_RELATIVE,
            folds=c1.INNER_FOLDS,
            seed=c1.SALTS["inner_folds"],
            dual=True,
        )

    def pooled(x, grid):
        return x if grid == 16 else pool_tokens(x.astype(np.float64), grid)

    out = {"grids": {}, "learning_curve": {}}
    for grid in GRIDS:
        xv, xh = pooled(visible, grid), pooled(hidden, grid)
        pv, ph = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
        for k in range(c1.OUTER_FOLDS):
            fit, held = fold != k, fold == k
            model = ridge(xv[fit], y[fit], roots[fit])
            pv[held], ph[held] = model.predict(xv[held]), model.predict(xh[held])
        ev = 100.0 * np.linalg.norm(pv - y, axis=1)
        eh = 100.0 * np.linalg.norm(ph - y, axis=1)
        out["grids"][f"{grid}x{grid}"] = {
            "visible": c1.median_ci(ev) | {"p87_5": float(np.percentile(ev, 87.5))},
            "plate_hidden": c1.median_ci(eh),
            "curve_mapped_of_32": float(32 * expected_rate(curve, ev).mean()),
        }
    rng = np.random.default_rng(SALTS["learning_curve"])
    for grid in (4, 8):
        xv = pooled(visible, grid)
        for frac in FRACTIONS:
            pv = np.full((n, 2), np.nan)
            for k in range(c1.OUTER_FOLDS):
                fit = np.flatnonzero(fold != k)
                fit = np.sort(rng.permutation(fit)[: int(round(frac * len(fit)))])
                held = fold == k
                pv[held] = ridge(xv[fit], y[fit], roots[fit]).predict(xv[held])
            e = 100.0 * np.linalg.norm(pv - y, axis=1)
            out["learning_curve"][f"{grid}x{grid} fit {frac}"] = {
                "fit_rows_per_fold": int(round(frac * int((fold != 0).sum()))),
                "median_cm": float(np.median(e)),
                "p87_5_cm": float(np.percentile(e, 87.5)),
            }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--c1-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: no CUDA context, no GPU lock
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    for name, want in PINNED.items():
        if sha256(args.c1_run / name) != want:
            raise SystemExit(f"{name} does not match the C1 record's sha256")
    report = json.loads((args.c1_run / "report.json").read_text())
    corpus = np.load(args.c1_run / "corpus.npz")
    vec, ok, arm = miss_vectors(report, corpus)
    miss = np.linalg.norm(vec, axis=1)
    curve = tolerance_curve(miss, ok)
    arm = np.array(arm)
    check = {
        name: {
            "measured_of_32": int(ok[arm == name].sum()),
            "curve_expected_of_32": float(32 * expected_rate(curve, miss[arm == name]).mean()),
        }
        for name in ("mean-proxy", "shuf-proxy", "N-proxy", "H-final")
    }
    out = {
        "ruling": "R15 (DRAFT), docs/experiments/apple_lewm_next_v2_direction.md §3",
        "development_only": True,
        "simulated_episodes": 0,
        "inputs_sha256": PINNED,
        "seeds_reset_values_only": list(SEEDS),
        "salts": SALTS,
        "attempts": int(len(ok)),
        "tolerance_curve": curve,
        "curve_check": check,
        "spreads": spreads(curve),
        "readouts": readouts(corpus, curve),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
