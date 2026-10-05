"""The C1-M feasibility record's runner (development only, before any protocol; no world model).

Ruling R15 (``docs/experiments/apple_lewm_next_v2_direction.md`` §5.3-§5.5); record
``docs/experiments/apple_lewm_next_v2_c1m_feasibility.md`` (declarations R16, §1); constants and
rules ``src/embodied_jepa/lewm_next_c1m.py``; workers ``src/embodied_jepa/lewm_next_c1m_runtime.py``
(on TASK-076's and C1's worker, hook and arms, unchanged). Guards from ``embodied_jepa.run_tools``
and TASK-076's harness module; this runner loads no other script.

One invocation runs every stage in the declared order and writes ``<output>/report.json`` (it
refuses to overwrite an existing output):

1. P-3's post-look estimates by G-repro (TASK-072 run-1's readout, refitted exactly).
2. **M-F1**: H-final(commit) on M1 (63000-63031) under the disc move at 3, 4, 5, 6 cm, stopping at
   the first radius below 30/32; the -y half-disc once if 3 cm fails; rho*.
3. **M-F2** at rho*: remaining motion and r from M-F1's attempts; the reach check (a_lo) on M1.
4. **M-F3** on F3 (63100-63131): the fresh ceiling and the four proxies. **Early stop** unless
   M-F1 to M-F3 pass (M-F4 to M-F7 are then reported as not run).
5. **M-F4** on T (63200-63231): tau_commit (H-final(commit) plus a planted error).
6. **M-F5a**: the 1 024-root corpus (63300-64323), then offline (CPU): cross-fitted pooled 4 x 4
   and 8 x 8 readouts at r, plate-hidden renders, the nested learning curve, R-plate at 405,
   H-sysid.
7. **M-F5b**: H-final(commit) on R (63132-63163); H-read on F3 + R, 4 x 4 (8 x 8 only if 4 x 4
   misses and its plate-hidden check passes).
8. **M-F6** on F3: H-rule, H-sysid (reading and true), H-now-reaim, H-rule-stale. **M-F7**: cost.

``--debug`` simulates debug seeds 64900-64999 only, at small sizes, runs every stage whatever a
check shows, and uses fixed stand-ins; nothing in it is read. It refuses uncommitted code (R16.11).
"""

from __future__ import annotations

import os

os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_next_c1 as c1  # noqa: E402
from embodied_jepa import lewm_next_c1m as c1m  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import plate_twin_v2 as pt  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import wm_critic_v2 as wc  # noqa: E402

TASK076_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-plate-twin-v2.json"
C1_FILES = c1m.C1_CODE_FILES  # C1's code, unchanged since 0c64ec2 (R16.1)
C1_REFERENCE = c1m.C1_REFERENCE
OWN_FILES = (
    "src/embodied_jepa/lewm_next_c1m.py",
    "src/embodied_jepa/lewm_next_c1m_runtime.py",
    "scripts/run_c1m_feasibility.py",
)
fpl.configure_headless()
GIB = 2**30
log = hz.log
GLOBAL_CAP_SECONDS = 4 * 3600.0
MAP_CAP_SECONDS = 3600.0
QUIET_POLL_SECONDS = 30.0
QUIET_WAIT_CAP_SECONDS = 4 * 3600.0
FEATURE_BATCH = 16


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on the C1-M worker; a dead worker or a cap is a GuardError; every
    map runs TASK-071's G-look check (as TASK-076's harness pool)."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import lewm_next_c1m_runtime as mrt

        super().__init__(workers, mrt.run_task, initializer=mrt.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = hz.check_look_states(out, self.look_reference)
        return out


# ----- preflight ----------------------------------------------------------------------------------
def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def check_code(report: dict) -> None:
    """C1's files are byte-identical to 0c64ec2, and this record's files are tracked (R16.1,
    R16.11)."""
    c1_hashes = {}
    for path in C1_FILES:
        want, got = git("rev-parse", f"{C1_REFERENCE}:{path}"), git("hash-object", path)
        if want != got:
            raise lp.GuardError(f"G-hash: {path} differs from {C1_REFERENCE}")
        c1_hashes[path] = got
    for path in OWN_FILES:
        try:
            git("ls-files", "--error-unmatch", path)
        except subprocess.CalledProcessError as error:
            raise lp.GuardError(f"G-hash: {path} is not committed") from error
    report["c1_code_git_blobs"] = c1_hashes
    report["own_code_git_blobs"] = {p: git("hash-object", p) for p in OWN_FILES}


def wait_quiet(report: dict, args) -> None:
    rule = wc.QUIET_MACHINE
    started = time.monotonic()
    while True:
        one, five, _ = os.getloadavg()
        if one <= rule["max_load_average_1min"] and five <= rule["max_load_average_5min"]:
            break
        if args.debug or time.monotonic() - started > QUIET_WAIT_CAP_SECONDS:
            if args.debug:
                break
            raise lp.GuardError(f"G-quiet: the machine stayed busy ({one:.2f}, {five:.2f})")
        log(f"waiting for a quiet machine (load {one:.2f}, {five:.2f})")
        time.sleep(QUIET_POLL_SECONDS)
    report["quiet_machine"] = {
        "rule": rule,
        "waited_seconds": time.monotonic() - started,
        "load_average_at_start": list(os.getloadavg()),
    }


def preflight(report: dict, args) -> dict:
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    rt.assert_local_import(ROOT, report)
    manifest = json.loads(TASK076_MANIFEST.read_text())
    report["task076_pins_at_preflight"] = hz.check_pins(manifest["hashes"])
    check_code(report)
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    hz.check_clean(dirty)  # the record run and every debug run (R16.11)
    report["platform"] = fpl.check_platform()
    report["thread_env"] = {k: os.environ.get(k) for k in pt.THREAD_ENV}
    if report["thread_env"] != pt.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {pt.THREAD_ENV}")
    wait_quiet(report, args)
    available = hz.mem_available_bytes()
    report["mem_available_at_start_gib"] = available / GIB
    if available < (pt.MEMORY["ceiling_gib"] + pt.MEMORY["headroom_gib"]) * GIB:
        raise lp.GuardError(f"G-memory: MemAvailable {available / GIB:.1f} GiB is too low")
    torch.set_num_threads(6)
    devices.configure_determinism("cuda", strict=True)  # no CUDA context is created
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise lp.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
    }
    digests = {
        "pretrained": pe.weights_digest(pe.load_pretrained()),
        "floor": pe.weights_digest(pe.random_init()),
    }
    if digests != manifest["encoder_digests"]:
        raise lp.GuardError("G-weights: an encoder digest differs from TASK-076's pin")
    report["encoder_digests"] = digests
    c1m.check_seed_ranges()
    found = hz.check_evidence(Path(args.evidence))
    report["evidence"] = {"root": str(args.evidence), "sha256": found["sha256"]}
    report["_run1"] = found["run1"]
    return manifest


# ----- helpers ------------------------------------------------------------------------------------
class Cohorts:
    """Seeds, resets and P-3's post-look estimates, rendered lazily per role (so an early stop
    simulates no seed it does not use)."""

    def __init__(self, args, pool, p_readout, encoder, report):
        self.debug = bool(args.debug)
        self.pool, self.p_readout, self.encoder, self.report = pool, p_readout, encoder, report
        self.resets: dict[int, dict] = {}
        self.est: dict[int, dict] = {}
        self.digests: dict[str, str] = {}

    def seeds(self, role: str) -> tuple[int, ...]:
        seeds = c1m.check_seeds(role, c1m.seeds_of(role, debug=self.debug), debug=self.debug)
        if role not in self.digests:
            resets = {s: c1m.reset_of(s) for s in seeds}
            self.resets |= resets
            self.digests[role] = lp.plan_digest({str(s): v for s, v in resets.items()})
            est = hz.cohort_estimates(
                self.pool, self.p_readout, self.encoder, seeds, resets, 1800.0
            )
            disagreements = est.pop("_render_disagreements", {})
            self.report["stages"].setdefault("render_disagreements", {})[role] = disagreements
            self.est |= est
            self.report["resets_digest"] = dict(self.digests)
        return seeds


def attempt_tasks(arm: str, seeds, co: Cohorts, move, per_seed=None, **extra) -> list[dict]:
    """``move``: (family, rho_cm) or None (no move; never in the record)."""
    tasks = []
    for s in seeds:
        offset = None if move is None else c1m.move_offset(s, *move).tolist()
        task = {
            "kind": "attempt",
            "arm": arm,
            "seed": s,
            "reset": co.resets[s],
            "estimates": co.est[s]["estimates"],
            "expected_frame_sha256": co.est[s]["frame_sha256"],
            "expected_state_sha256": co.est[s]["state_sha256"],
            "wall_seconds": pt.CAPS_SECONDS["per_attempt"],
            "move_offset": offset,
        }
        tasks.append(task | (per_seed(s) if per_seed else {}) | extra)
    return tasks


def run_arm(pool, tasks, what: str) -> list[dict]:
    records = pool.map(tasks, MAP_CAP_SECONDS, what)
    arm = tasks[0]["arm"]
    if arm not in c1m.PRIVILEGED_ARMS:
        for r in records:
            if r.get("blocked") is None and not r.get("privileged_ok", False):
                raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")
    log(f"{what}: {sum(bool(r['success']) for r in records)}/{len(records)}")
    return records


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def lean(records, *, keep_path: bool = False) -> list[dict]:
    out = []
    for item in hz.strip(records):
        item = dict(item)
        item.pop("captured", None)
        if not keep_path and "kpred" in item:
            item["kpred"] = {k: v for k, v in item["kpred"].items() if k != "plate_path"}
        out.append(item)
    return out


def arm_summary(records) -> dict:
    misses = [
        r["commit"]["landing_miss_cm"]
        for r in records
        if r.get("commit") and r["commit"].get("landing_miss_cm") is not None
    ]
    a_values = [r["commit"]["a_true"] for r in records if r.get("commit")]
    grasp = [r.get("first_grasp_step") for r in records]

    def stats(v):
        if not v:
            return {"median": None, "p87_5": None, "max": None}
        return {
            "median": float(np.median(v)),
            "p87_5": float(np.percentile(v, 87.5)),
            "max": float(np.max(v)),
        }

    return {
        "count": sum(successes(records)),
        "n": len(records),
        "per_reset": successes(records),
        "refused": sum(r["blocked"] is not None for r in records),
        "move_blocked": sum(r.get("termination_reason") == "move_blocked" for r in records),
        "fallbacks": sum(bool(r.get("commit", {}).get("fallback")) for r in records),
        "grasp_latched_before_move": sum(g is not None and int(g) < c1m.MOVE_STEP for g in grasp),
        "landing_miss_cm": stats(misses),
        "a_true": {
            "median": float(np.median(a_values)) if a_values else None,
            "min": float(np.min(a_values)) if a_values else None,
            "max": float(np.max(a_values)) if a_values else None,
        },
        "seconds": {
            "median": float(np.median([r["seconds"] for r in records])),
            "max": float(np.max([r["seconds"] for r in records])),
        },
    }


def move_summary(seeds, family: str, rho_cm: float) -> dict:
    offsets = np.asarray([c1m.move_offset(s, family, rho_cm) for s in seeds])
    norms = 100.0 * np.linalg.norm(offsets, axis=1)
    return {
        "family": family,
        "rho_cm": rho_cm,
        "redraws": {str(s): c1m.move_uniforms(s)["k"] for s in seeds if c1m.move_uniforms(s)["k"]},
        "offset_cm": {"median": float(np.median(norms)), "max": float(np.max(norms))},
    }


# ----- M-F1 and M-F2 ------------------------------------------------------------------------------
def stage_m1(report, pool, co: Cohorts, args):
    seeds = co.seeds("M1")
    families, chosen, records_at = {}, None, {}
    for family in c1m.FAMILIES:
        counts, per = {}, {}
        for rho in c1m.RHO_GRID_CM:
            records = run_arm(
                pool, attempt_tasks("H-final", seeds, co, (family, rho)), f"M-F1 {family} {rho} cm"
            )
            counts[rho] = sum(successes(records))
            records_at[(family, rho)] = records
            per[str(rho)] = {
                "arm": arm_summary(records),
                "move": move_summary(seeds, family, rho),
                "attempts": lean(records, keep_path=True),
            }
            if counts[rho] < c1m.CEILING_MIN and not args.debug:
                break
            if args.debug:
                break  # debug: one radius per family is enough mechanics
        star = c1m.rho_star(counts)
        families[family] = {"counts": {str(k): v for k, v in counts.items()}, "rho_star": star}
        families[family]["radii"] = per
        if star is not None:
            chosen = (family, star)
            break
        if args.debug and family == "disc":
            continue
    report["stages"]["m_f1"] = {"seeds": list(seeds), "families": families}
    if chosen is None and args.debug:
        chosen = (c1m.DEBUG_STANDINS["family"], c1m.DEBUG_STANDINS["rho_cm"])
        report["stages"]["m_f1"]["debug_standin"] = list(chosen)
    report["stages"]["m_f1"]["chosen"] = None if chosen is None else list(chosen)
    return chosen, (records_at.get(chosen) if chosen else None)


def motion_and_r(ceiling_records):
    s1 = c1.RULE["s1"]
    distances, remaining, speeds, m2 = [], [], [], []
    for r in ceiling_records:
        k = r["kpred"]
        path = {int(t): np.asarray(v) for t, v in k["plate_path"].items()}
        if r["blocked"] is not None or s1 not in path:
            distances.append(None)
        else:
            distances.append(
                {t: 100.0 * float(np.linalg.norm(v - path[s1])) for t, v in path.items()}
            )
            remaining.append(k["remaining_405_cm"])
        if k.get("palm_speed_405_cm") is not None:
            speeds.append(k["palm_speed_405_cm"])
        if k.get("m2_405_cm") is not None:
            m2.append(k["m2_405_cm"])
    r_step = c1.read_step(distances)
    within = {
        str(t): sum(1 for d in distances if d is not None and d.get(t, math.inf) <= 0.1)
        for t in range(c1.RULE["s0"], s1 + 1)
    }
    return remaining, r_step, speeds, m2, within


def stage_m2(report, pool, co: Cohorts, chosen, ceiling_records, args):
    seeds = co.seeds("M1")
    remaining, r_step, speeds, m2, within = motion_and_r(ceiling_records)
    complete, per_level = {}, {}
    for a in c1.A_LEVELS:
        ok = np.ones(len(seeds), bool)
        detail = {}
        for b in c1.REACH_B_M:
            tasks = attempt_tasks("reach", seeds, co, chosen, a=a, b_m=b)
            records = run_arm(pool, tasks, f"M-F2 reach a={a} b={b * 100:+.0f} cm")
            done = np.array(
                [
                    r["blocked"] is None
                    and not r.get("commit", {}).get("fallback", True)
                    and r["executed_steps"] is not None
                    and int(r["executed_steps"]) >= c1.TRANSFER_END
                    for r in records
                ]
            )
            ok &= done
            detail[f"{b:+.2f}"] = {
                "complete": int(done.sum()),
                "success": sum(successes(records)),
                "incomplete_seeds": [s for s, d in zip(seeds, done, strict=True) if not d],
                "terminations": sorted({str(r["termination_reason"]) for r in records}),
            }
        complete[a] = int(ok.sum())
        per_level[str(a)] = {"complete_all_three": int(ok.sum()), "by_b": detail}
        if args.debug:
            break
    a_lo = None if args.debug else c1.decide_a_lo(complete)
    f2 = c1m.decide_f2(remaining, r_step, speeds, a_lo)
    report["stages"]["m_f2"] = {
        "chosen": list(chosen),
        "remaining_405_cm": remaining,
        "palm_speed_405_cm": speeds,
        "m2_405_cm": {
            "median": float(np.median(m2)) if m2 else None,
            "max": float(np.max(m2)) if m2 else None,
        },
        "within_0_1_cm_count_by_step": within,
        "reach": {"levels": per_level, "a_lo": a_lo},
        "f2": f2,
    }
    if args.debug:
        a_lo = c1m.DEBUG_STANDINS["a_lo"]
        r_step = r_step if r_step is not None else c1m.DEBUG_STANDINS["r"]
        report["stages"]["m_f2"]["debug_standins"] = {"a_lo": a_lo, "r": r_step}
    return f2, a_lo, r_step


# ----- M-F3 ---------------------------------------------------------------------------------------
def stage_m3(report, pool, co: Cohorts, chosen, a_lo):
    seeds = co.seeds("F3")
    n = len(seeds)
    family, rho = chosen
    moved = {s: c1m.moved_plate(s, family, rho).tolist() for s in seeds}
    p_bar = c1m.plate_mean(family, rho).tolist()
    arms = {"H-final": run_arm(pool, attempt_tasks("H-final", seeds, co, chosen), "M-F3 ceiling")}
    for arm in c1m.PROXY_ARMS:
        per = None
        if arm == "shuf-proxy":

            def per(s):
                return {"foreign_plate": moved[seeds[c1.foreign_index(seeds.index(s), n)]]}

        extra = {"a_lo": a_lo} | ({"plate_mean": p_bar} if arm == "mean-proxy" else {})
        arms[arm] = run_arm(pool, attempt_tasks(arm, seeds, co, chosen, per, **extra), arm)
    for s, r in zip(seeds, arms["H-final"], strict=True):  # the hook's plate at 405 is the plan's
        if r.get("commit") and not np.allclose(r["commit"]["p"], moved[s], atol=1e-9):
            raise lp.GuardError(f"G-move: seed {s} plate at 405 is not its moved plate")
    ceiling = successes(arms["H-final"])
    f3 = c1m.decide_f3(ceiling, {a: successes(arms[a]) for a in c1m.PROXY_ARMS})
    report["stages"]["m_f3"] = {
        "seeds": list(seeds),
        "chosen": list(chosen),
        "a_lo": a_lo,
        "a_n": c1.n_proxy_a(a_lo),
        "plate_mean": p_bar,
        "move": move_summary(seeds, family, rho),
        "arms": {a: arm_summary(r) for a, r in arms.items()},
        "f3": f3,
        "attempts": {a: lean(r) for a, r in arms.items()},
    }
    return f3, arms["H-final"]


# ----- M-F4 ---------------------------------------------------------------------------------------
def stage_m4(report, pool, co: Cohorts, chosen):
    seeds = co.seeds("T")
    counts, arms = {}, {}
    for level in c1m.TAU_LEVELS_CM:
        tasks = attempt_tasks(
            "H-final-planted",
            seeds,
            co,
            chosen,
            lambda s, level=level: {"planted_m": c1m.planted_error_m(s, level)},
        )
        records = run_arm(pool, tasks, f"M-F4 planted {level} cm")
        counts[level] = sum(successes(records))
        arms[str(level)] = records
    tau = c1m.decide_tau(counts)
    report["stages"]["m_f4"] = {
        "seeds": list(seeds),
        "move": move_summary(seeds, *chosen),
        "tau": tau,
        "arms": {k: arm_summary(r) for k, r in arms.items()},
        "attempts": {k: lean(r) for k, r in arms.items()},
    }
    return tau, arms


# ----- M-F5a: the corpus and the offline readouts -------------------------------------------------
def stage_corpus(report, pool, co: Cohorts, chosen, a_lo, r_step, folder: Path):
    seeds = co.seeds("corpus")
    draws = {s: c1m.corpus_aim(s, a_lo) for s in seeds}
    tasks = attempt_tasks(
        "collect",
        seeds,
        co,
        chosen,
        lambda s: {"a": draws[s][0], "b_m": draws[s][1]},
        capture=[c1.COMMIT_STEP, int(r_step)],
    )
    records = run_arm(pool, tasks, "M-F5a corpus")
    rows = [
        r
        for r in records
        if r.get("commit") is not None
        and str(c1.COMMIT_STEP) in r["captured"]
        and str(r_step) in r["captured"]
    ]
    s1 = c1.RULE["s1"]
    arrays = {
        "seeds": np.asarray([r["seed"] for r in rows], np.int64),
        "a": np.asarray([draws[r["seed"]][0] for r in rows]),
        "b_m": np.asarray([draws[r["seed"]][1] for r in rows]),
        "p405": np.asarray([r["commit"]["p"] for r in rows]),
        "h405": np.asarray([r["commit"]["h"] for r in rows]),
        "target": np.asarray([r["commit"]["target"] for r in rows]),
        "fallback": np.asarray([r["commit"]["fallback"] for r in rows]),
        "plate_r": np.asarray([r["captured"][str(r_step)]["plate"] for r in rows]),
        "plate_s1": np.asarray(
            [r["kpred"]["plate_path"].get(str(s1), [np.nan, np.nan]) for r in rows]
        ),
        "success": np.asarray([bool(r["success"]) for r in rows]),
        "frames405": np.stack([r["captured"][str(c1.COMMIT_STEP)]["visible"] for r in rows]),
        "frames_r": np.stack([r["captured"][str(r_step)]["visible"] for r in rows]),
        "hidden_r": np.stack([r["captured"][str(r_step)]["hidden"] for r in rows]),
        "r": np.asarray(int(r_step)),
    }
    path = folder / "corpus.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(path, **arrays)
    report["stages"]["m_f5a_corpus"] = {
        "seeds": [int(seeds[0]), int(seeds[-1])],
        "roots": len(records),
        "rows": len(rows),
        "excluded": sorted(set(seeds) - {r["seed"] for r in rows}),
        "count": sum(successes(records)),
        "fallbacks": int(arrays["fallback"].sum()),
        "move": move_summary(seeds, *chosen),
        "path": str(path),
        "sha256": hz.sha256_file(path),
        "attempts": lean(records),
    }
    return arrays


def featurise(encoder, frames, *, full: bool):
    """Pooled 4 x 4 and 8 x 8 latents (float32) and, if ``full``, the full tokens (float32), from
    ``pretrained_encoder.features`` (the closed-loop featurisation), CPU, in batches."""
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    n = len(frames)
    out = {g: np.empty((n, g * g * 384), np.float32) for g in c1m.GRIDS}
    tokens_all = np.empty((n, 256 * 384), np.float32) if full else None
    for lo in range(0, n, FEATURE_BATCH):
        tokens = pe.features(encoder, np.asarray(frames[lo : lo + FEATURE_BATCH]), batch=16)[
            "tokens"
        ]
        for g in c1m.GRIDS:
            out[g][lo : lo + len(tokens)] = pool_tokens(tokens, g)
        if full:
            tokens_all[lo : lo + len(tokens)] = tokens
        del tokens
    return out, tokens_all


def ridge(x, y, groups):
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        x,
        y,
        groups,
        lambdas=c1m.LAMBDA_GRID_RELATIVE,
        folds=c1m.INNER_FOLDS,
        seed=c1m.SALTS["inner_folds"],
        dual=True,
    )


def stage_offline(report, arrays: dict, folder: Path, tau_cm):
    """CPU only: frozen DINOv2 features, the cross-fitted readouts at r on both grids, the hidden
    renders, the nested learning curve, R-plate at 405, H-sysid and the closed-loop readouts."""
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.plate_twin_v2_offline import save_readout

    started = time.monotonic()
    encoder = pe.load_pretrained()
    _, full405 = featurise(encoder, arrays["frames405"], full=True)
    pooled_r, _ = featurise(encoder, arrays["frames_r"], full=False)
    hidden_r, _ = featurise(encoder, arrays["hidden_r"], full=False)
    feat_seconds = time.monotonic() - started
    n = len(arrays["seeds"])
    roots = arrays["seeds"].astype(str)
    fold = c1m.outer_folds(n)
    y405, yr = arrays["p405"], arrays["plate_r"]
    grids, curves, selections = {}, {}, []
    for g in c1m.GRIDS:
        pv, ph = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
        curve_pred = {f: np.full((n, 2), np.nan) for f in c1m.FRACTIONS}
        sizes = {f: [] for f in c1m.FRACTIONS}
        for k in range(c1m.OUTER_FOLDS):
            fit, held = np.flatnonzero(fold != k), fold == k
            subsets = c1m.nested_subsets(fit, k)
            if not np.array_equal(subsets[1.0], fit):
                raise lp.GuardError("G-curve: the full-data subset is not the fold's fit rows")
            for f in c1m.FRACTIONS:
                rows = subsets[f]
                model = ridge(pooled_r[g][rows], yr[rows], roots[rows])
                curve_pred[f][held] = model.predict(pooled_r[g][held])
                sizes[f].append(int(len(rows)))
                if f == 1.0:
                    pv[held] = curve_pred[f][held]
                    ph[held] = model.predict(hidden_r[g][held])
                    selections.append({"grid": g, "fold": k, "selection": model.selection})
        for name, values in (("visible", pv), ("hidden", ph), *curve_pred.items()):
            if not np.isfinite(values).all():
                raise lp.GuardError(f"G-finite: grid {g} {name} lacks an out-of-fold prediction")
        ev = 100.0 * np.linalg.norm(pv - yr, axis=1)
        eh = 100.0 * np.linalg.norm(ph - yr, axis=1)
        e_curve = {f: 100.0 * np.linalg.norm(curve_pred[f] - yr, axis=1) for f in c1m.FRACTIONS}
        guard = c1m.median_difference_ci(e_curve[0.75], e_curve[1.0])
        hidden_stats = c1m.median_ci(eh) | {"p87_5": float(np.percentile(eh, 87.5))}
        grids[g] = {
            "visible": c1m.median_ci(ev) | {"p87_5": float(np.percentile(ev, 87.5))},
            "plate_hidden": hidden_stats,
            "hidden_ok": c1m.hidden_ok(hidden_stats, tau_cm),
        }
        curves[g] = {
            "fit_rows_per_fold": {str(f): sizes[f] for f in c1m.FRACTIONS},
            "median_cm": {str(f): float(np.median(e_curve[f])) for f in c1m.FRACTIONS},
            "p87_5_cm": {str(f): float(np.percentile(e_curve[f], 87.5)) for f in c1m.FRACTIONS},
            "guard_3_4_minus_all": guard,
            "still_falling": c1m.still_falling(guard),
        }
    # R-plate (full tokens) at 405, the constant prior at r, H-sysid's residuals (reported)
    p405 = np.full((n, 2), np.nan)
    prior = np.full((n, 2), np.nan)
    sysid = {k: np.full((n, 2), np.nan) for k in ("true", "reading")}
    for k in range(c1m.OUTER_FOLDS):
        fit, held = fold != k, fold == k
        model = ridge(full405[fit], y405[fit], roots[fit])
        p405[held] = model.predict(full405[held])
        prior[held] = yr[fit].mean(axis=0)
        selections.append({"grid": "full405", "fold": k, "selection": model.selection})
    for k in range(c1m.OUTER_FOLDS):
        fit, held = fold != k, fold == k
        h, g = arrays["h405"], arrays["target"]
        coef = c1.sysid_fit(y405[fit], h[fit], g[fit], yr[fit])
        sysid["true"][held] = c1.sysid_predict(coef, y405[held], h[held], g[held])
        coef = c1.sysid_fit(p405[fit], h[fit], g[fit], yr[fit])
        sysid["reading"][held] = c1.sysid_predict(coef, p405[held], h[held], g[held])
    for name, values in {"p405": p405, "prior": prior, **sysid}.items():
        if not np.isfinite(values).all():
            raise lp.GuardError(f"G-finite: {name} lacks an out-of-fold prediction")

    def err(pred, y):
        e = 100.0 * np.linalg.norm(pred - y, axis=1)
        return c1m.median_ci(e) | {"p87_5": float(np.percentile(e, 87.5))}

    reported = {
        "r_plate_405": err(p405, y405),
        "prior_r": err(prior, yr),
        "sysid_true": err(sysid["true"], yr),
        "sysid_reading": err(sysid["reading"], yr),
        "plate_motion_405_to_r_cm": c1m.median_ci(100.0 * np.linalg.norm(yr - y405, axis=1)),
    }
    readouts = {}
    for g in c1m.GRIDS:  # the closed-loop H-read readouts, fitted on every corpus row
        model = ridge(pooled_r[g], yr, roots)
        path = folder / f"r_read_{g}.npz"
        readouts[f"r_read_{g}"] = {
            "path": str(path),
            "sha256": save_readout(path, model),
            "selection": model.selection,
        }
    model = ridge(full405, y405, roots)
    path = folder / "r_plate.npz"
    readouts["r_plate"] = {
        "path": str(path),
        "sha256": save_readout(path, model),
        "selection": model.selection,
    }
    coef_true = c1.sysid_fit(y405, arrays["h405"], arrays["target"], yr)
    coef_read = c1.sysid_fit(p405, arrays["h405"], arrays["target"], yr)
    report["stages"]["m_f5a_offline"] = {
        "device": "cpu",
        "featurisation_seconds": feat_seconds,
        "rows": n,
        "folds": fold.tolist(),
        "tau_commit_cm": tau_cm,
        "grids": {str(g): v for g, v in grids.items()},
        "learning_curve": {str(g): v for g, v in curves.items()},
        "reported": reported,
        "selections": selections,
        "readouts": readouts,
        "sysid": {"coef_true": coef_true.tolist(), "coef_reading": coef_read.tolist()},
        "seconds": time.monotonic() - started,
    }
    config = {}
    for name, item in readouts.items():
        config[name], config[f"{name}_sha256"] = item["path"], item["sha256"]
    return grids, curves, config, coef_true, coef_read


# ----- M-F5b, M-F6, M-F7 --------------------------------------------------------------------------
def stage_m5b(report, pool, co: Cohorts, chosen, r_step, f3_ceiling, grids):
    f3_seeds, r_seeds = co.seeds("F3"), co.seeds("R")
    r_ceiling = run_arm(pool, attempt_tasks("H-final", r_seeds, co, chosen), "M-F5b ceiling on R")
    ceiling_records = list(f3_ceiling) + r_ceiling
    seeds = (*f3_seeds, *r_seeds)
    ceiling = successes(ceiling_records)
    reads, arms = {}, {}
    for g in c1m.GRIDS:
        if g != c1m.GRIDS[0]:
            if any(reads[x]["passes"] for x in reads):
                break
            if not grids[g]["hidden_ok"]:
                reads_note = f"{g}x{g} not admitted: its plate-hidden check did not pass"
                report["stages"].setdefault("m_f5b", {})["not_admitted"] = reads_note
                break
        tasks = attempt_tasks(
            "H-read", seeds, co, chosen, readout=f"r_read_{g}", grid=g, read_step=int(r_step)
        )
        records = run_arm(pool, tasks, f"M-F5b H-read {g}x{g}")
        reads[g] = c1m.decide_read(ceiling, successes(records))
        errors = [
            it["reading_error_cm"]
            for r in records
            for d in r["decisions"]
            for it in d.get("lookahead", {}).get("iterations", [])
            if "reading_error_cm" in it
        ]
        reads[g]["reading_error_cm"] = c1m.median_ci(errors) if errors else None
        reads[g]["converged"] = sum(
            bool(d.get("lookahead", {}).get("converged")) for r in records for d in r["decisions"]
        )
        arms[g] = records
    report["stages"].setdefault("m_f5b", {}).update(
        {
            "seeds": [int(seeds[0]), int(seeds[-1])],
            "ceiling": arm_summary(ceiling_records),
            "reads": {str(g): v for g, v in reads.items()},
            "arms": {str(g): arm_summary(r) for g, r in arms.items()},
            "attempts": {"ceiling_R": lean(r_ceiling)}
            | {f"H-read {g}x{g}": lean(r) for g, r in arms.items()},
        }
    )
    return reads, ceiling_records, arms


def stage_m6(report, pool, co: Cohorts, chosen, a_lo, coef_true, coef_read, f3_ceiling):
    seeds = co.seeds("F3")
    arms = {}
    for arm in c1m.COMPARATOR_ARMS:
        extra = {"a_lo": a_lo}
        if arm == "H-sysid":
            extra["sysid_coef"] = coef_read.tolist()
        elif arm == "H-sysid-true":
            extra["sysid_coef"] = coef_true.tolist()
        arms[arm] = run_arm(pool, attempt_tasks(arm, seeds, co, chosen, **extra), f"M-F6 {arm}")
    reading_err = [
        100.0 * float(np.linalg.norm(np.asarray(r["decisions"][0]["reading"]) - r["commit"]["p"]))
        for arm in ("H-rule", "H-sysid")
        for r in arms[arm]
        if r["decisions"] and r.get("commit") and r["decisions"][0].get("reading") is not None
    ]
    ceiling = successes(f3_ceiling)
    counts = {a: sum(successes(r)) for a, r in arms.items()}
    best = max(("H-rule", "H-sysid"), key=lambda a: counts[a])
    report["stages"]["m_f6"] = {
        "arms": {a: arm_summary(r) for a, r in arms.items()},
        "paired_vs_ceiling": {
            a: c1m.paired_interval(ceiling, successes(r)) for a, r in arms.items()
        },
        "reading_error_405_cm": c1m.median_ci(reading_err) if reading_err else None,
        "best_on_reading": {
            "arm": best,
            "count": counts[best],
            "tie": counts["H-rule"] == counts["H-sysid"],
        },
        "attempts": {a: lean(r) for a, r in arms.items()},
    }


def stage_m7(report, ceiling_records, read_arms) -> dict:
    seconds = [r["seconds"] for r in ceiling_records] + [
        r["seconds"] for records in read_arms.values() for r in records
    ]
    worst = float(np.max(seconds))
    out = {
        "attempts": len(seconds),
        "max_seconds": worst,
        "median_seconds": float(np.median(seconds)),
        "passes": worst <= c1m.COST_SECONDS_MAX,
        "consequence": None if worst <= c1m.COST_SECONDS_MAX else "the per-attempt cap is reviewed",
    }
    report["stages"]["m_f7"] = out
    return out


# ----- the run ------------------------------------------------------------------------------------
def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "record": c1m.RECORD,
        "ruling": c1m.RULING,
        "declaring_commit": c1m.DECLARING_COMMIT,
        "debug": bool(args.debug),
        "outcome": None,
        "stages": {},
        "not_run": [],
        "argv": sys.argv[1:],
        "paths": {"output": str(output.resolve()), "evidence": str(Path(args.evidence).resolve())},
        "started_utc": hz.utc(),
        "no_world_model": True,
        "gpu": "none: every stage, featurisation included, runs on the CPU",
        "salts": c1m.SALTS,
        "seed_ranges": c1m.DEBUG_RANGES if args.debug else c1m.SEED_RANGES,
    }
    clock = hz.Clock(GLOBAL_CAP_SECONDS)
    watch = rt.MemoryWatch(
        pt.MEMORY["ceiling_gib"] * GIB, pt.MEMORY["sample_seconds"], measure=pt.MEMORY["measure"]
    )
    guards = rt.install_guards(watch=watch, log=log)
    report["_watch"] = watch
    pool = None
    try:
        manifest = preflight(report, args)
        config = {
            "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence)),
            "torch_threads": pt.WORKER_TORCH_THREADS,
        }
        workers = int(args.workers or pt.SIM_WORKERS)
        report["workers"] = workers
        pool = Pool(workers, config)
        p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
        report["first_render_utc"] = hz.utc()
        co = Cohorts(args, pool, p_readout, encoder, report)
        checks = {"rho": None, "f2": None, "f3": None}
        clock.check("M-F1")
        chosen, m1_records = stage_m1(report, pool, co, args)
        decision = None
        if chosen is not None:
            checks["rho"] = chosen[1]
            clock.check("M-F2")
            f2, a_lo, r_step = stage_m2(report, pool, co, chosen, m1_records, args)
            checks["f2"] = f2
            if f2["passes"] or args.debug:
                clock.check("M-F3")
                f3, f3_ceiling = stage_m3(report, pool, co, chosen, a_lo)
                checks["f3"] = f3
        early = c1m.verdict(rho=checks["rho"], f2=checks["f2"], f3=checks["f3"])
        if early["row"] in ("M-INFEASIBLE", "M-TWINS-NONE", "M-TWINS-ESCALATE") and not args.debug:
            decision = early
            report["stopped"] = (
                f"early stop after M-F1 to M-F3 ({early['row']}): M-F4 to M-F7 not run (R15.7)"
            )
            report["not_run"] = ["M-F4", "M-F5a", "M-F5b", "M-F6", "M-F7"]
            if checks["f3"] is None:
                report["not_run"].insert(0, "M-F3")
            if checks["f2"] is None:
                report["not_run"].insert(0, "M-F2")
        else:
            report["early_verdict"] = early
            clock.check("M-F4")
            tau, _ = stage_m4(report, pool, co, chosen)
            tau_cm = tau["tau_commit_cm"]
            if args.debug and tau_cm is None:
                tau_cm = c1m.DEBUG_STANDINS["tau_commit_cm"]
            clock.check("M-F5a corpus")
            arrays = stage_corpus(report, pool, co, chosen, a_lo, r_step, output)
            report["pool_close_1"] = pool.close()
            pool = None
            clock.check("M-F5a offline")
            grids, curves, readout_config, coef_true, coef_read = stage_offline(
                report, arrays, output, tau_cm
            )
            del arrays
            pool = Pool(workers, config | readout_config)
            co.pool = pool
            clock.check("M-F5b")
            reads, ceiling_records, read_arms = stage_m5b(
                report, pool, co, chosen, r_step, f3_ceiling, grids
            )
            clock.check("M-F6")
            stage_m6(report, pool, co, chosen, a_lo, coef_true, coef_read, f3_ceiling)
            stage_m7(report, ceiling_records + list(m1_records or []), read_arms)
            decision = c1m.verdict(
                rho=checks["rho"],
                f2=checks["f2"],
                f3=checks["f3"],
                tau=tau,
                hidden4_ok=grids[4]["hidden_ok"],
                reads=reads,
                falling={g: curves[g]["still_falling"] for g in c1m.GRIDS},
            )
        report["checks"] = {
            "M-F1": {"rho_star": checks["rho"], "chosen": report["stages"]["m_f1"]["chosen"]},
            "M-F2": checks["f2"],
            "M-F3": None
            if checks["f3"] is None
            else {k: checks["f3"][k] for k in ("ceiling", "ceiling_ok", "twins_none", "passes")},
        }
        report["verdict"] = decision
        report["pinned_at_end"] = hz.check_pins(manifest["hashes"])
        report["revision_at_end"] = hz.revision()
        if hz.tracked_tree_dirty():
            raise lp.GuardError("G-hash: the tracked tree changed during the run")
        report["outcome"] = "C1M-DEBUG" if args.debug else decision["row"]
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        guards.void(report, error)
    finally:
        guards.finish(report, pool)
        if pool is not None:
            report["pool_close"] = pool.close_record
        report.pop("_watch", None)
        report.pop("_run1", None)
        report["ended_utc"] = hz.utc()
        report["total_seconds"] = clock.elapsed()
        hz.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence", required=True, help="TASK-072 run-1's evidence root")
    parser.add_argument("--debug", action="store_true", help="debug seeds only; nothing is read")
    parser.add_argument("--workers", type=int, default=None, help="debug only (1-6)")
    args = parser.parse_args(argv)
    if args.workers is not None and (not args.debug or not 1 <= args.workers <= 6):
        parser.error("--workers is for debug runs only, from 1 to 6")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
