"""TASK-092 D-RATE: one development-only closed-loop reading (DECISIONS (ba), R27.100-R27.106;
protocol docs/experiments/grasp_rate_aware_wm_v1.md §17). Not gated, not Stage BD, no claim; its
counts are development readings, not results.

It runs TASK-092's frozen closed-loop code path (``scripts/run_task092.py`` ``goals`` and ``run``,
the path Stage BD uses) from a run tree that must be a clean tree at ``64e87f0`` (``--run-tree``
or ``DRATE_RUN_TREE``), and changes no frozen module. The only addition is one development
cohort, ``dev-drate`` = resets **90810-90841** (32), registered in the imported modules at run
time, which admits exactly two runs: M-req-87100 (Stage T's G-req checkpoint) and M-v1-87100
(TASK-087's G). Its name does not start with ``debug``, so the runner's non-debug guards apply
unchanged: clean tracked tree, the 7 200 s cap, the exact seed set, the goals' revision, and the
once-only ledger (``outputs/task092-ledger``). The planner, λ, the CEM settings and the
checkpoints are the frozen ones. Every report carries a ``development`` field naming this
script and its sha256.

* ``goals --output DIR`` and ``run --arm {M-req,M-v1} --goals DIR --output DIR``: the runner's own
  subcommands on ``dev-drate`` (``--model-seed 87100`` and ``--workers 6`` are added, as in BD).
  Each is one job under ``scripts/gpu_run.sh --wait --min-free-gib 8 --board``.
* ``summary --runs DIR --t-summary FILE``: R27.102's reading (CPU): successes, grasp contact,
  maximum rise, the executed-close timing and the clipped share for both arms, and the row
  D-MOVES / D-FLAT / D-WORSE from the paired difference, in that declared order.

    T=/home/huhn/develop/emai/worktrees/task092-drate
    MUJOCO_GL=egl DRATE_RUN_TREE=$T scripts/gpu_run.sh --wait --min-free-gib 8 --board -- \\
        $T/.venv/bin/python scripts/dev_task092_drate.py goals --output <evidence>
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
COHORT = "dev-drate"
SEEDS = range(90810, 90842)
RUNS = (("M-req", 87100), ("M-v1", 87100))
WORKERS = 6
MOVES, WORSE = 4, -4  # R27.102: D-MOVES if diff >= +4; D-WORSE if <= -4; else D-FLAT (|diff| <= 3)

sys.path.insert(0, str(RUN_TREE / "scripts"))
sys.path.insert(0, str(RUN_TREE / "src"))
import run_task092 as run  # noqa: E402  (the frozen runner; it puts RUN_TREE/src first)

from embodied_jepa import grasp_rate_aware as ra  # noqa: E402

SELF_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# ----- the one addition: the development cohort ------------------------------------------------
ra.COHORTS[COHORT] = SEEDS
ra.GRASP_COHORTS = (*ra.GRASP_COHORTS, COHORT)
_check_run = ra.check_run


def _check_run_drate(cohort, arm, model_seed, *, debug_req_seed=None):
    if cohort == COHORT and (arm, model_seed) not in RUNS:
        raise ra.ContractError(f"{arm}-{model_seed} is not a run of {COHORT}")
    return _check_run(cohort, arm, model_seed, debug_req_seed=debug_req_seed)


ra.check_run = _check_run_drate
_base_report = run.base_report


def _base_report_drate(cohort=None, **kw):
    report = _base_report(cohort, **kw)
    report["development"] = {
        "name": "D-RATE",
        "rulings": "DECISIONS (ba) R27.100-R27.106",
        "gated": False,
        "script": "scripts/dev_task092_drate.py",
        "script_sha256": SELF_SHA256,
    }
    return report


run.base_report = _base_report_drate


def _guard_tree() -> None:
    head = run.git("rev-parse", "HEAD")
    if not head.startswith(REVISION):
        raise run.rt.GuardError(f"G-revision: the run tree is at {head}, not {REVISION}")
    if run.git("status", "--porcelain", "--untracked-files=no"):
        raise run.rt.GuardError("G-clean: the run tree is dirty")


# ----- summary (R27.102) -----------------------------------------------------------------------
def summary(runs_dir: Path, t_summary: Path) -> dict:
    sys.path.insert(0, str(RUN_TREE / "scripts"))
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
    for name, r in runs.items():
        if [e["seed"] for e in r["episodes"]] != list(SEEDS):
            problems.append(f"{name}: not the cohort's seeds")
        if not str(r.get("revision", "")).startswith(REVISION):
            problems.append(f"{name}: revision {r.get('revision')}")
        if r.get("goals_sha256") != run.sha(runs_dir / f"goals-{COHORT}.json"):
            problems.append(f"{name}: goals differ")
        if (r.get("development") or {}).get("name") != "D-RATE":
            problems.append(f"{name}: not a D-RATE report")
    req = runs.get("M-req-87100", {})
    want = t["models"]["G-req-87100"]["checkpoint_sha256"]
    if (req.get("model") or {}).get("checkpoint_sha256") != want:
        problems.append("M-req-87100: not Stage T's G-req checkpoint")
    out = {
        "name": "D-RATE",
        "development_only": True,
        "gated": False,
        "cohort": COHORT,
        "seeds": [SEEDS.start, SEEDS.stop - 1],
        "problems": problems,
        "targets": st._count([x["target"] for x in goals["goals"]]),
        "rejected_draws": goals["rejections"],
    }
    if problems or len(runs) != 2:
        out["row"] = None
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


def main(argv=None) -> int:
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
    _guard_tree()
    if argv[:1] == ["goals"]:
        return run.main(argv + ["--cohort", COHORT, "--workers", "8"])
    if argv[:1] == ["run"]:
        return run.main(
            argv + ["--cohort", COHORT, "--model-seed", "87100", "--workers", str(WORKERS)]
        )
    raise SystemExit("usage: dev_task092_drate.py {goals,run,summary} ...")


if __name__ == "__main__":
    raise SystemExit(main())
