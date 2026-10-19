"""TASK-101 amendment A2 (protocol docs/experiments/so101_closed_loop_lift_v1.md §19): measure
the pose-error tolerance envelope of the privileged follower v1 on ``so101_cube_lift_v1`` and
apply §19.4's rule for the switching tolerance τ_s.

CPU and MuJoCo only, rendering off; development seeds E 102000-102031 (reserve 102032-102047).
For each seed: the unperturbed goal demonstration (§4's rejection rule, replacement from the
reserve) gives the reference joints at commands 40 and 70; then, for each variant ("pre": subgoal
1; "carry": subgoal 2, the rule's reading; "reaim": subgoal 2, reported only), a δ = 0 run that
must execute exactly the demonstration's commands (guard G-zero), and one run per declared level.
Refuses an existing output and a dirty tracked tree.

    MUJOCO_GL=egl uv run --no-sync python scripts/measure_task101_envelope.py \
        --output outputs/task101-envelope
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_closed_loop_runtime as rt
from embodied_jepa import so101_switch_envelope as env

PINNED = (
    "src/embodied_jepa/so101_switch_envelope.py",
    "scripts/measure_task101_envelope.py",
    "src/embodied_jepa/so101_followers.py",
    "src/embodied_jepa/so101_scripted.py",
    "src/embodied_jepa/so101_tasks.py",
    "src/embodied_jepa/so101_tasks_lift_v1.py",
    "src/embodied_jepa/so101_simulation.py",
    "src/embodied_jepa/so101.py",
    "configs/so101_sim_action.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_variant(seed: int, variant: str, delta) -> tuple[dict, np.ndarray]:
    robot = rt.make_robot(False)
    rec = rt.Recorder(robot)
    task = rt.lift_task(robot)
    try:
        out = env.perturbed_lift(rec, task, seed, variant=variant, delta=delta)
    except RuntimeError as error:  # the follower raises on a stopped execution
        out = {"success": False, "termination_reason": f"stopped: {error}", "stopped": True}
    finally:
        robot.close()
    return out, np.asarray(rec.applied, np.float32).reshape(-1, cl.JOINTS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    report = rt.header("envelope", envelope_protocol="so101_closed_loop_lift_v1.md §19")
    if report["tracked_tree_dirty"]:
        raise SystemExit("refusing: the tracked tree is dirty")
    report["files_sha256"] = {rel: sha(rt.ROOT / rel) for rel in PINNED}
    report["declared"] = {
        "levels_rad": list(env.LEVELS_RAD),
        "per_level": env.PER_LEVEL,
        "pass_count": env.PASS_COUNT,
        "salt": env.SALT,
        "cohort": [env.ENVELOPE.first, env.ENVELOPE.count, list(env.ENVELOPE.reserve)],
        "handover_command": env.HANDOVER_COMMAND,
        "rule_variants": list(env.RULE_VARIANTS),
        "reported_variants": list(env.REPORTED_VARIANTS),
    }
    out.mkdir(parents=True)
    t0 = time.monotonic()

    # §4's rejection rule on the unperturbed demonstration; replacement from the reserve.
    kept, rejected, demos = [], [], {}
    spare = iter(env.ENVELOPE.spare())
    for seed in env.ENVELOPE.main():
        while True:
            env.check_seed(seed)
            d = rt.demonstrate(seed, render=False)
            demos[seed] = d
            if d["kept"]:
                kept.append(seed)
                break
            rejected.append({"seed": seed, "reason": d["reason"]})
            seed = next(spare, None)
            if seed is None:
                break
    void = None if len(kept) == env.PER_LEVEL else f"kept {len(kept)} of {env.PER_LEVEL}"
    rt.log(f"demonstrations: kept {len(kept)}, rejected {len(rejected)}")

    episodes, counts = [], {}
    applied_store: dict[str, np.ndarray] = {}
    for seed in kept:
        d = demos[seed]
        ref = {c["command"]: np.asarray(c["joints"], float) for c in d["captured"]}
        for variant, subgoal in env.VARIANTS:
            command = env.HANDOVER_COMMAND[subgoal]
            # Guard G-zero: δ = 0 reproduces the demonstration's executed commands exactly.
            zero, applied = run_variant(seed, variant, np.zeros(env.ARM_JOINTS))
            same = applied.shape == d["applied"].shape and np.array_equal(applied, d["applied"])
            if not same or bool(zero.get("success")) != bool(d["success"]):
                void = void or f"G-zero failed: seed {seed} variant {variant}"
            episodes.append(
                {"seed": seed, "variant": variant, "level": 0.0, "guard_zero_identical": same}
            )
            for level in env.LEVELS_RAD:
                delta = env.offset(seed, subgoal, level)
                r, applied = run_variant(seed, variant, delta)
                hand = r.get("handover_joints")
                achieved = (
                    None
                    if hand is None
                    else env.rms(
                        np.asarray(hand)[: env.ARM_JOINTS] - ref[command][: env.ARM_JOINTS]
                    )
                )
                key = f"{seed}/{variant}/{level}"
                applied_store[key] = applied
                score = r.get("score") or {}
                episodes.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "subgoal": subgoal,
                        "level": level,
                        "delta": delta.tolist(),
                        "success": bool(r.get("success")),
                        "termination_reason": r.get("termination_reason"),
                        "success_step": score.get("success_step"),
                        "commands": int(len(applied)),
                        "handover_arm_rms_rad": achieved,
                        "handover_jaw_error_rad": (
                            None if hand is None else float(hand[cl.JAW] - ref[command][cl.JAW])
                        ),
                        "ended_before_handover": bool(r.get("ended_before_handover", True)),
                        "applied_sha256": hashlib.sha256(applied.tobytes()).hexdigest(),
                    }
                )
                counts[(variant, level)] = counts.get((variant, level), 0) + int(r.get("success"))
        rt.log(f"seed {seed} done")

    for variant, _ in env.VARIANTS:
        for level in env.LEVELS_RAD:
            counts.setdefault((variant, level), 0)
    if rt.tree_dirty() or rt.git("rev-parse", "HEAD") != report["revision"]:
        void = void or "G-tree: the tracked tree changed during the run"
    row = env.envelope_row(
        {k: v for k, v in counts.items() if k[0] in env.RULE_VARIANTS}, len(kept), void
    )
    table = []
    for variant, subgoal in env.VARIANTS:
        for level in env.LEVELS_RAD:
            k = counts[(variant, level)]
            n = len(kept)
            lo, hi = env.clopper_pearson(k, n) if n else (None, None)
            achieved = [
                e["handover_arm_rms_rad"]
                for e in episodes
                if e["variant"] == variant
                and e["level"] == level
                and e.get("handover_arm_rms_rad") is not None
            ]
            table.append(
                {
                    "variant": variant,
                    "subgoal": subgoal,
                    "level_rad": level,
                    "successes": k,
                    "of": n,
                    "ci95": [lo, hi],
                    "rule_reading": variant in env.RULE_VARIANTS,
                    "handover_rms_median_rad": float(np.median(achieved)) if achieved else None,
                }
            )
    np.savez_compressed(out / "applied_actions.npz", **applied_store)
    report.update(
        {
            "kept": kept,
            "rejected": rejected,
            "demonstrations": {
                str(s): {
                    k: demos[s][k]
                    for k in ("success", "termination_reason", "success_step", "commands", "kept")
                }
                for s in demos
            },
            "table": table,
            "row": row,
            "episodes": episodes,
            "seconds": time.monotonic() - t0,
            "finished_utc": rt.utc(),
            "tracked_tree_dirty_at_end": rt.tree_dirty(),
            "revision_at_end": rt.git("rev-parse", "HEAD"),
        }
    )
    digest = rt.write_json(out / "envelope.json", report)
    print(json.dumps({"row": row, "envelope_sha256": digest}, indent=1))
    for t in table:
        print(
            f"{t['variant']:6s} sg{t['subgoal']} {t['level_rad']:.2f} rad: "
            f"{t['successes']}/{t['of']}  [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]"
            f"  handover rms {t['handover_rms_median_rad']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
