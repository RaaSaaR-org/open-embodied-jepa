"""TASK-081 simulation workers (protocol ``docs/experiments/apple_lewm_commit_precision_v2.md``
§3, §4): one closed-loop attempt of any arm.

TASK-080's worker (``lewm_pr_v2_runtime``) is imported and reused unchanged: its initializer, its
``PredReadoutAim`` (the encoded 405 frame, the p̂-and-h grid, the stand-in chunks, each arm's own
readout, R-S or R-N), its controllers for every non-candidate arm and its ``run_attempt_task``.
What changes is only the step after the grid, selected per task through ``lewm_pr_v2_runtime``'s
module-level ``choose``, which this worker replaces **for the duration of one task** and restores
afterwards (:func:`run_task`):

* **affine_local** (§3.1) for W, N, L-shuf and L-mean: :func:`choose_affine_local`, the grid and
  its argmin exactly as ``lewm_c1m_v2_runtime.choose_from_grid`` computes them, then the design
  note's ``commit_precision_dev.affine_fixed_point`` over ``local_mask``'s neighbourhood, with its
  fallbacks to the argmin; one logged roll-out of the committed aim, which never changes it;
* **frozen** for W-frozen: TASK-077's ``choose_from_grid`` unchanged (TASK-080's W, run as arm
  "W", so it reads W's model, start latent and R-S), relabelled "W-frozen" in its record;
* **L-rand** draws a feasible candidate with this task's salt 8303, no solver.

Every candidate attempt logs its solver and the runner refuses a record whose solver is not the
arm's declared one (G-solver). Who reads what follows TASK-071's rule (G-privileged), as in
TASK-080. NumPy at import.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import commit_precision_dev as cpd
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_runtime as wrt
from embodied_jepa import lewm_cp_v2 as cpv
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_pr_v2_runtime as prt


def worker_init(config: dict) -> None:
    """TASK-080's worker initializer unchanged (R-S and R-N with their sha256, R8 for H-read)."""
    prt.worker_init(config)


def _grid(p_hat, h, targets, chunks, feasible, predict):
    """``choose_from_grid``'s grid step: the feasible candidates' predicted plates in one batch and
    the lowest |p~(g) - g| (ties to the lowest row-major index)."""
    index = np.flatnonzero(np.asarray(feasible, bool))
    chunks = np.asarray(chunks)
    plates = np.asarray(predict(chunks[index]), np.float64).reshape(-1, 2)
    scores = np.linalg.norm(plates - targets[index], axis=1)
    j = int(np.argmin(scores))
    return index, plates, scores, j


def choose_affine_local(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of, *,
                        a_lo: float = lm.A_LO):  # fmt: skip
    """§3.1: the grid and its argmin (unchanged), then the clipped fixed point of the least-squares
    affine fit p~(g) ~ c + J g over the feasible candidates within 2 a-steps and 2 cm in b of the
    argmin (``commit_precision_dev``), with the argmin as the fallback. Returns (g, log)."""
    p_hat = np.asarray(p_hat, np.float64).reshape(2)
    h = np.asarray(h, np.float64).reshape(2)
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    feasible = np.asarray(feasible, bool)
    log: dict = {"arm": arm, "solver": "affine_local", "feasible": int(feasible.sum()),
                 "candidates": int(len(targets))}  # fmt: skip
    if not feasible.any():
        log |= {"fallback_all_infeasible": True, "clipped": False, "rollouts": 0}
        return p_hat.copy(), log
    log["fallback_all_infeasible"] = False
    index, plates, scores, j = _grid(p_hat, h, targets, chunks, feasible, predict)
    best = int(index[j])
    g0 = targets[best].copy()
    mask = cpd.local_mask(lm.GRID, best)[index]
    points = int(mask.sum())
    g, jac = cpd.affine_fixed_point(targets[index][mask], plates[mask], p_hat=p_hat, h=h,
                                    a_lo=a_lo)  # fmt: skip
    reason, unclipped = None, None
    design = np.hstack([targets[index][mask], np.ones((points, 1))])
    rank = int(np.linalg.matrix_rank(design)) if points else 0  # reported (a rank below 3 is
    # a collinear neighbourhood, where lstsq returns the minimum-norm fit; the solver is unchanged)
    if jac is None:
        reason = "fewer_than_4_points"
    elif g is None:
        reason = "near_singular"
    else:
        coef, *_ = np.linalg.lstsq(design, plates[mask], rcond=None)
        unclipped = np.linalg.solve(np.eye(2) - coef[:2].T, coef[2])
    commands_g, ok = (None, False) if g is None else chunk_of(g)
    if g is not None and not ok:
        reason = "infeasible_chunk"
    fallback = reason is not None
    if fallback:
        g = g0.copy()
        commands_g, _ok = chunk_of(g)
    plate = np.asarray(predict(np.asarray(commands_g)[None]), np.float64).reshape(2)
    residual = c1.clip_to_box(plate, p_hat, h, a_lo) - g
    eig = None if jac is None else np.linalg.eigvals(np.asarray(jac, np.float64))
    log |= {
        "grid_index": best,
        "grid_best": list(lm.GRID[best]),
        "grid_best_score_cm": 100.0 * float(scores[j]),
        "points": points,
        "design_rank": rank,
        "jacobian": None if jac is None else np.asarray(jac).tolist(),
        "eigenvalues_real": None if eig is None else [float(v) for v in np.real(eig)],
        "eigenvalues_complex": None if eig is None else bool(np.any(np.abs(np.imag(eig)) > 0)),
        "fallback_grid_argmin": fallback,
        "fallback_reason": reason,
        "unclipped": None if unclipped is None else unclipped.tolist(),
        "clipped": bool(
            not fallback and unclipped is not None and np.linalg.norm(unclipped - g) > 1e-12
        ),
        "residual_cm": 100.0 * float(np.linalg.norm(residual)),
        "rollouts": int(len(index)) + 1,
        "converged": None,
    }
    return np.asarray(g, np.float64).copy(), log


def choose_l_rand(arm, p_hat, targets, feasible, *, seed: int):
    """L-rand with salt 8303: TASK-080's L-rand branch, this task's salt."""
    p_hat = np.asarray(p_hat, np.float64).reshape(2)
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    feasible = np.asarray(feasible, bool)
    log = {"arm": arm, "solver": "none", "feasible": int(feasible.sum()),
           "candidates": int(len(targets))}  # fmt: skip
    if not feasible.any():
        log |= {"fallback_all_infeasible": True, "clipped": False, "rollouts": 0}
        return p_hat.copy(), log
    i = cpv.l_rand_index(seed, feasible)
    log |= {"fallback_all_infeasible": False, "grid_index": i, "grid_best": list(cpv.GRID[i]),
            "clipped": False, "rollouts": 0}  # fmt: skip
    return targets[i].copy(), log


def make_choose(solver: str):
    """The replacement for ``lewm_pr_v2_runtime.choose`` during one task of ``solver``."""
    if solver not in ("affine_local", "frozen", "none"):
        raise lp.GuardError(f"G-solver: unknown solver {solver!r}")

    def choose(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of, *, tolerance_m, seed):
        if arm == "L-rand":
            if solver != "none":
                raise lp.GuardError("G-solver: L-rand has no solver")
            return choose_l_rand(arm, p_hat, targets, feasible, seed=seed)
        if solver == "frozen":
            g, log = wrt.choose_from_grid(
                arm, p_hat, h, targets, chunks, feasible, predict, chunk_of,
                tolerance_m=tolerance_m, seed=seed,
            )  # fmt: skip
            log["solver"] = "frozen"
            return g, log
        if solver == "affine_local":
            return choose_affine_local(arm, p_hat, h, targets, chunks, feasible, predict,
                                       chunk_of)  # fmt: skip
        raise lp.GuardError(f"G-solver: {arm} cannot use solver {solver!r}")

    choose.solver = solver
    return choose


def logged_solver(record: dict):
    """The solver a candidate attempt's first decision logged (None before 405)."""
    decisions = record.get("decisions") or []
    if not decisions:
        return None
    return (decisions[0].get("world_model") or {}).get("solver")


def run_attempt_task(task: dict) -> dict:
    """One attempt: candidate arms run TASK-080's ``run_attempt_task`` with this task's solver in
    place of ``choose`` for this task only; every other arm runs it unchanged."""
    arm = task["arm"]
    if arm not in cpv.SOLVER_OF:
        if arm not in cpv.S_ARMS:
            raise lp.GuardError(f"not a TASK-081 arm: {arm}")
        return prt.run_attempt_task(task)
    solver = cpv.SOLVER_OF[arm]
    inner = dict(task, arm=cpv.MODEL_ARM.get(arm, arm))
    original = prt.choose
    prt.choose = make_choose(solver)
    try:
        out = prt.run_attempt_task(inner)
    finally:
        prt.choose = original
    out["arm"] = arm
    out["solver"] = solver
    got = logged_solver(out)
    if got is not None and got != solver:
        raise lp.GuardError(f"G-solver: {arm} logged {got!r}, not {solver!r}")
    return out


def run_task(task: dict) -> dict:
    if task["kind"] == "attempt":
        return run_attempt_task(task)
    return prt.run_task(task)  # frames and G-repro, as TASK-080's worker
