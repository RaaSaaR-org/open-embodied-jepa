"""TASK-092 D-RATE′: one development-only closed-loop reading (DECISIONS (ba), R27.100-R27.106,
and (bb), R27.109; protocol docs/experiments/grasp_rate_aware_wm_v1.md §17-§18). Not gated, not
Stage BD, no claim; its counts are development readings, not results. D-RATE itself (cohort
``dev-drate``, resets 90810-90841) ended D-VOID before any episode (R27.107); its resets are spent.

It runs TASK-092's frozen closed-loop code path (``scripts/run_task092.py`` ``goals`` and ``run``,
the path Stage BD uses) from a run tree that must be a clean tree at ``64e87f0``
(``DRATE_RUN_TREE``), and changes no frozen module. The only addition is one development cohort,
``dev-drate2`` = resets **90850-90881** (32), registered in the imported modules at run time, which
admits exactly two runs: M-req-87100 (Stage T's G-req checkpoint) and M-v1-87100 (TASK-087's G).
Its name does not start with ``debug``, so the runner's non-debug guards apply unchanged: clean
tracked tree, the 7 200 s cap, the exact seed set, the goals' revision, and the once-only ledger
(``outputs/task092-ledger``). The planner, λ, the CEM settings and the checkpoints are the frozen
ones. Every report carries a ``development`` field naming this script, its sha256 and revision.

Two guards added after D-RATE's D-VOID (R27.109):

* **Run tree as working directory.** ``goals`` and ``run`` change into the run tree first (the
  frozen LeWM loader resolves ``third_party/le-wm`` relative to the working directory) and refuse
  to start if ``third_party/le-wm`` is not a directory there.
* **Preflight.** Before the runner is called, and so before any ledger key is claimed, both models
  (M-req-87100 and M-v1-87100) and the encoder are loaded on CUDA in a spawned process (as the
  runner's workers load them); if either fails, the command refuses to start.

* ``goals --output DIR`` and ``run --arm {M-req,M-v1} --goals DIR --output DIR``: the runner's own
  subcommands on ``dev-drate2`` (``--model-seed 87100`` and ``--workers 6`` are added, as in BD).
  Each is one job under ``scripts/gpu_run.sh --wait --min-free-gib 8 --board``.
* ``summary --runs DIR --t-summary FILE``: R27.102-R27.103's reading (CPU): successes, grasp
  contact, maximum rise, the executed-close timing and the clipped share for both arms, and the row
  D-VOID / D-MOVES / D-WORSE / D-FLAT from the paired difference, in that declared order.

    T=/home/huhn/develop/emai/worktrees/task092-drate
    cd $T && MUJOCO_GL=egl DRATE_RUN_TREE=$T scripts/gpu_run.sh --wait --min-free-gib 8 --board \\
        -- $T/.venv/bin/python <checkout>/scripts/dev_task092_drate.py goals --output <evidence>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

RUN_TREE = Path(os.environ.get("DRATE_RUN_TREE", "/home/huhn/develop/emai/worktrees/task092-drate"))
REVISION = "64e87f0"
NAME = "D-RATE′"
COHORT = "dev-drate2"
SEEDS = range(90850, 90882)
SPENT = range(90810, 90842)  # D-RATE's resets (R27.108); never used again
RUNS = (("M-req", 87100), ("M-v1", 87100))
FAMILIES = {"M-req": "G-req", "M-v1": "G-v1"}  # ra.CLOSED_LOOP_ARMS
WORKERS = 6
MOVES, WORSE = 4, -4  # R27.102: D-MOVES if diff >= +4; D-WORSE if <= -4; else D-FLAT (|diff| <= 3)
LEWM_SOURCE = Path("third_party") / "le-wm"  # as models/lewm.py resolves it (relative to cwd)

SELF_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class Refused(SystemExit):
    """A guard of this script refused to start; nothing was claimed."""


def _main_revision() -> str | None:
    """The revision of the checkout that holds this script (not the run tree's)."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ----- guards added after D-RATE's D-VOID (R27.109) ---------------------------------------------
def enter_run_tree(run_tree: Path = RUN_TREE) -> Path:
    """Make the run tree the working directory (inherited by the runner's spawned workers) and
    refuse unless the LeWM source resolves there as the frozen loader resolves it."""
    tree = Path(run_tree).resolve()
    if not tree.is_dir():
        raise Refused(f"G-cwd: run tree {tree} does not exist")
    os.chdir(tree)
    if Path.cwd().resolve() != tree or not LEWM_SOURCE.is_dir():
        raise Refused(f"G-cwd: {LEWM_SOURCE} is not a directory under the run tree {tree}")
    return tree


def _load_in_child(family: str, seed: int) -> dict:
    """Load one model and the encoder exactly as the runner's workers do (spawned child)."""
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware_runtime as rr

    devices.configure_determinism("cuda", strict=False)
    _, info = rr.load_family(family, seed, "cuda")
    gr.encoder("cuda")
    return {"checkpoint_sha256": (info or {}).get("checkpoint_sha256")}


def spawned_load(family: str, seed: int) -> dict:
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(_load_in_child, (family, seed))


def preflight(load=spawned_load, runs=RUNS) -> dict:
    """Load every model of the reading; refuse to start if any load fails (R27.109)."""
    loaded, failed = {}, {}
    for arm, seed in runs:
        try:
            loaded[f"{arm}-{seed}"] = load(FAMILIES[arm], seed)
        except Exception as error:  # noqa: BLE001
            failed[f"{arm}-{seed}"] = f"{type(error).__name__}: {error}"
    if failed:
        raise Refused(f"G-preflight: model load failed, nothing claimed: {failed}")
    print(json.dumps({"preflight": loaded}), flush=True)
    return loaded


# ----- the frozen runner, with the one addition (the development cohort) ----------------------
def install(run_tree: Path = RUN_TREE):
    """Import the frozen runner from the run tree and register the development cohort."""
    sys.path.insert(0, str(Path(run_tree) / "scripts"))
    sys.path.insert(0, str(Path(run_tree) / "src"))
    import run_task092 as run  # the frozen runner; it puts RUN_TREE/src first

    from embodied_jepa import grasp_rate_aware as ra

    if getattr(ra, "_drate_installed", False):
        return run, ra
    if set(SEEDS) & set(SPENT):
        raise Refused("D-RATE's spent resets are not reused (R27.108)")
    ra.COHORTS[COHORT] = SEEDS
    ra.GRASP_COHORTS = (*ra.GRASP_COHORTS, COHORT)
    check_run = ra.check_run

    def check_run_drate(cohort, arm, model_seed, *, debug_req_seed=None):
        if cohort == COHORT and (arm, model_seed) not in RUNS:
            raise ra.ContractError(f"{arm}-{model_seed} is not a run of {COHORT}")
        return check_run(cohort, arm, model_seed, debug_req_seed=debug_req_seed)

    ra.check_run = check_run_drate
    base_report = run.base_report

    def base_report_drate(cohort=None, **kw):
        report = base_report(cohort, **kw)
        report["development"] = {
            "name": NAME,
            "rulings": "DECISIONS (ba) R27.100-R27.106, (bb) R27.109",
            "gated": False,
            "script": "scripts/dev_task092_drate.py",
            "script_sha256": SELF_SHA256,
            "script_revision": _main_revision(),
        }
        return report

    run.base_report = base_report_drate
    ra._drate_installed = True
    return run, ra


def _guard_tree(run) -> None:
    head = run.git("rev-parse", "HEAD")
    if not head.startswith(REVISION):
        raise run.rt.GuardError(f"G-revision: the run tree is at {head}, not {REVISION}")
    if run.git("status", "--porcelain", "--untracked-files=no"):
        raise run.rt.GuardError("G-clean: the run tree is dirty")


# ----- summary (R27.102-R27.103) ----------------------------------------------------------------
def summary(runs_dir: Path, t_summary: Path) -> dict:
    run, _ = install()
    import summarize_task092 as st

    allr, _ = st._runs(runs_dir, COHORT)
    runs = {k: v for k, v in allr.items() if not v.get("determinism_rerun")}
    names = [f"{a}-{s}" for a, s in RUNS]
    problems = []
    if sorted(runs) != sorted(names):
        problems.append(f"runs {sorted(runs)} are not {names}")
    problems += st.common_checks(list(runs.values()), COHORT, len(SEEDS))
    goals = st.load(runs_dir / f"goals-{COHORT}.json")
    t = st.load(t_summary)
    if t.get("row") != "T-DONE":
        problems.append("Stage T did not end T-DONE")
    for name, r in runs.items():
        if [e["seed"] for e in r["episodes"]] != list(SEEDS):
            problems.append(f"{name}: not the cohort's seeds")
        if not str(r.get("revision", "")).startswith(REVISION):
            problems.append(f"{name}: revision {r.get('revision')}")
        if r.get("goals_sha256") != run.sha(runs_dir / f"goals-{COHORT}.json"):
            problems.append(f"{name}: goals differ")
        if (r.get("development") or {}).get("name") != NAME:
            problems.append(f"{name}: not a {NAME} report")
    req = runs.get("M-req-87100", {})
    want = t.get("models", {}).get("G-req-87100", {}).get("checkpoint_sha256")
    if want is None or (req.get("model") or {}).get("checkpoint_sha256") != want:
        problems.append("M-req-87100: not Stage T's G-req checkpoint")
    out = {
        "name": NAME,
        "development_only": True,
        "gated": False,
        "cohort": COHORT,
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "problems": problems,
        "targets": st._count([x["target"] for x in goals["goals"]]),
        "rejected_draws": goals["rejections"],
    }
    if problems or len(runs) != 2:
        out["row"] = "D-VOID"  # §17.3 rule 1
        return out
    out["arms"] = st._arm_table(runs)
    out["deferral_and_rate"] = st._deferral_and_rate(runs)
    a = {e["seed"]: bool(e["success"]) for e in runs["M-req-87100"]["episodes"]}
    b = {e["seed"]: bool(e["success"]) for e in runs["M-v1-87100"]["episodes"]}
    k_req, k_v1 = sum(a.values()), sum(b.values())
    diff = k_req - k_v1
    out |= {
        "k_req": k_req,
        "k_v1": k_v1,
        "difference": diff,
        "discordant_req_only": sum(a[s] and not b[s] for s in SEEDS),
        "discordant_v1_only": sum(b[s] and not a[s] for s in SEEDS),
        "row": "D-MOVES" if diff >= MOVES else ("D-WORSE" if diff <= WORSE else "D-FLAT"),
        "revision": runs["M-req-87100"]["revision"],
    }
    return out


def main(argv=None, *, run_tree: Path = RUN_TREE, load=spawned_load, runner=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["summary"]:
        p = argparse.ArgumentParser()
        p.add_argument("summary")
        p.add_argument("--runs", required=True)
        p.add_argument("--t-summary", required=True)
        a = p.parse_args(argv)
        path = Path(a.runs) / f"summary-{COHORT}.json"
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        s = summary(Path(a.runs), Path(a.t_summary))
        path.write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
        print(json.dumps({k: s.get(k) for k in ("row", "k_req", "k_v1", "problems")}))
        return 0
    if argv[:1] not in (["goals"], ["run"]):
        raise SystemExit("usage: dev_task092_drate.py {goals,run,summary} ...")
    enter_run_tree(run_tree)  # G-cwd, before anything is imported from the run tree
    if runner is None:
        run, _ = install(run_tree)
        _guard_tree(run)
        runner = run.main
    preflight(load)  # G-preflight, before the runner claims any ledger key
    if argv[:1] == ["goals"]:
        return runner(argv + ["--cohort", COHORT, "--workers", "8"])
    return runner(argv + ["--cohort", COHORT, "--model-seed", "87100", "--workers", str(WORKERS)])


if __name__ == "__main__":
    raise SystemExit(main())
