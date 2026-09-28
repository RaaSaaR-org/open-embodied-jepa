"""Compare the Isaac Sim G1+Dex3 joint dump with the project's MuJoCo embodiment.

Development spike helper (not a gated experiment). Run on the host with the project
environment; it reads the JSON written by ``bringup_g1_dex3.py`` inside the container:

    uv run --no-sync python scripts/isaac/compare_joints.py \
        --isaac outputs/isaac-bringup-spike-<n>/run/isaac_joints.json \
        --output outputs/isaac-bringup-spike-<n>/joint_comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mujoco_joints() -> dict[str, dict]:
    from embodied_jepa.simulation import MuJoCoSimulation

    sim = MuJoCoSimulation(render=False)
    model = sim.model
    out = {}
    for order, (name, jid) in enumerate(zip(sim.joint_names, sim.joint_ids, strict=True)):
        lo, hi = (float(v) for v in model.jnt_range[int(jid)])
        out[name] = {"order": order, "lower": lo, "upper": hi}
    return out


def group(name: str) -> str:
    if "_hand_" in name:
        return "dex3"
    if any(k in name for k in ("shoulder", "elbow", "wrist")):
        return "arm"
    if name.startswith("waist"):
        return "waist"
    return "leg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-3, help="limit tolerance [rad]")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    isaac = {j["name"]: j for j in json.loads(args.isaac.read_text())["joints"]}
    mj = mujoco_joints()
    only_mj = sorted(set(mj) - set(isaac))
    only_isaac = sorted(set(isaac) - set(mj))
    rows = []
    for name in sorted(set(mj) & set(isaac), key=lambda n: mj[n]["order"]):
        a, b = mj[name], isaac[name]
        dl, du = b["lower"] - a["lower"], b["upper"] - a["upper"]
        rows.append(
            {
                "name": name,
                "group": group(name),
                "mujoco": [a["lower"], a["upper"]],
                "isaac": [b["lower"], b["upper"]],
                "limit_match": abs(dl) <= args.tol and abs(du) <= args.tol,
                "max_abs_limit_diff_rad": max(abs(dl), abs(du)),
            }
        )
    counts = {
        g: {
            "mujoco": sum(group(n) == g for n in mj),
            "isaac": sum(group(n) == g for n in isaac),
        }
        for g in ("leg", "waist", "arm", "dex3")
    }
    mismatched = [r for r in rows if not r["limit_match"]]
    summary = {
        "mujoco_joint_count": len(mj),
        "isaac_joint_count": len(isaac),
        "counts_by_group": counts,
        "only_in_mujoco": only_mj,
        "only_in_isaac": only_isaac,
        "common": len(rows),
        "limit_mismatches": [
            {k: r[k] for k in ("name", "group", "mujoco", "isaac", "max_abs_limit_diff_rad")}
            for r in mismatched
        ],
        "arm_dex3_all_names_and_limits_match": not any(
            group(n) in ("arm", "dex3") for n in only_mj + only_isaac
        )
        and not any(r["group"] in ("arm", "dex3") for r in mismatched),
        "tolerance_rad": args.tol,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for r in mismatched:
        print(
            f"  {r['name']:32s} mj=[{r['mujoco'][0]:+.4f},{r['mujoco'][1]:+.4f}] "
            f"isaac=[{r['isaac'][0]:+.4f},{r['isaac'][1]:+.4f}]"
        )
    assert all(math.isfinite(r["max_abs_limit_diff_rad"]) for r in rows)


if __name__ == "__main__":
    main()
