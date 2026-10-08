"""TASK-081 design note, development only: W's commit precision under C1-M (not gated).

The design note ``docs/experiments/apple_lewm_commit_precision_v2_design.md`` asks why TASK-080's
W stopped at its 10-refinement cap on 17 of 64 gated resets, and whether another solver for the
same fixed point, on the same predictor and readout, converges. This module holds

* the **solver variants** as pure functions of a predictor (no simulator, no torch). Every variant
  starts from the frozen controller's grid argmin, so a variant changes only what happens after
  the grid: ``frozen`` (TASK-077's refinement, cap 10), ``cap30`` (the same, cap 30), ``damped``
  (step 1/2, cap 10), ``bestres`` (the frozen path, but at the cap the evaluated point with the
  smallest residual), ``affine_local`` / ``affine_global`` (the fixed point of a least-squares
  affine fit to W's own grid predictions, near the argmin or over the whole grid);
* a **worker** for the development closed loop (``scripts/dev_commit_precision.py``): TASK-080's
  worker unchanged, with W's ``choose`` wrapped so that every variant's aim is computed at 405
  from the same roll-outs and logged, and the task's ``variant`` is the aim executed.

Nothing here is a gated arm, nothing is frozen, and no TASK-076/077/080 module is edited (the
wrapper replaces ``lewm_pr_v2_runtime.choose`` inside a development worker process only). The
frozen variant is checked against TASK-077's ``choose_from_grid`` on every call. NumPy at import.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_next_c1 as c1

VARIANTS = ("frozen", "cap30", "damped", "bestres", "affine_local", "affine_global")
EXECUTABLE = ("frozen", "damped", "affine_local")  # the variants the closed loop executes
DAMPING = 0.5  # declared before any development run: contracts for local slopes in (-3, 1)
DAMPED_CAP = 10  # the frozen cap, so "damped" changes only the step
LONG_CAP = 30
LOCAL_A_STEPS = 2  # affine_local: grid points within 2 a-steps and 2 cm in b of the grid argmin
LOCAL_B_M = 0.02


def _residual(plate, g, p_hat, h, a_lo) -> np.ndarray:
    """clip(p~(g)) - g: the frozen controller's move from g (its logged ``gap`` is the norm)."""
    return c1.clip_to_box(plate, p_hat, h, a_lo) - np.asarray(g, np.float64)


def iterate(g0, single, feasible, *, alpha, cap, tolerance_m, p_hat, h, a_lo, known=()):
    """TASK-077's refinement form with step ``alpha``: at each iteration roll g out once, take
    r = clip(p~(g)) - g, move to g + alpha r if its stand-in chunk is feasible (else stop, not
    converged), and stop as converged once |r| <= tolerance. At the cap the last move is committed
    unevaluated, as ``choose_from_grid`` does; ``alpha = 1`` is that controller exactly.

    ``single(g) -> plate [2]`` (one roll-out of g's chunk); ``feasible(g) -> bool``. ``known`` is a
    sequence of (g, plate) already evaluated in this order, reused while the path matches. Returns
    (aim, converged, evaluated [(g, r)], stopped_infeasible, roll-outs made by this call)."""
    g = np.asarray(g0, np.float64).copy()
    evaluated, converged, stopped, rollouts = [], False, False, 0
    known = list(known)
    for k in range(int(cap)):
        if k < len(known) and np.array_equal(np.asarray(known[k][0], np.float64), g):
            plate = np.asarray(known[k][1], np.float64)
        else:
            plate = np.asarray(single(g), np.float64)
            rollouts += 1
        r = _residual(plate, g, p_hat, h, a_lo)
        evaluated.append((g.copy(), r))
        nxt = g + float(alpha) * r
        if not feasible(nxt):
            stopped = True
            break
        g = nxt
        if np.linalg.norm(r) <= tolerance_m:
            converged = True
            break
    return g, converged, evaluated, stopped, rollouts


def best_residual(evaluated) -> tuple[np.ndarray, float, int]:
    """The evaluated point with the smallest residual (no extra roll-out)."""
    norms = [float(np.linalg.norm(r)) for _g, r in evaluated]
    i = int(np.argmin(norms))
    return evaluated[i][0].copy(), norms[i], i


def affine_fixed_point(targets, plates, *, p_hat, h, a_lo):
    """Least squares p~(g) ~ c + J g over the candidates given, solved for g = c + J g, clipped.
    Returns (g, J); g is None when the fit is under-determined or I - J is near singular."""
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    plates = np.asarray(plates, np.float64).reshape(-1, 2)
    if len(targets) < 4:
        return None, None
    x = np.hstack([targets, np.ones((len(targets), 1))])
    coef, *_ = np.linalg.lstsq(x, plates, rcond=None)
    jac, c = coef[:2].T, coef[2]
    m = np.eye(2) - jac
    if abs(np.linalg.det(m)) < 1e-6:
        return None, jac
    return c1.clip_to_box(np.linalg.solve(m, c), p_hat, h, a_lo), jac


def local_mask(grid, best: int) -> np.ndarray:
    grid = np.asarray(grid, np.float64)
    a0, b0 = grid[best]
    return (np.abs(grid[:, 0] - a0) <= LOCAL_A_STEPS * c1.A_STEP + 1e-9) & (
        np.abs(grid[:, 1] - b0) <= LOCAL_B_M + 1e-9
    )


def all_variants(p_hat, h, targets, feasible, grid_plates, single, feasible_aim, *, known,
                 tolerance_m, a_lo, grid=lm.GRID):  # fmt: skip
    """Every variant's aim from the same grid prediction.

    ``grid_plates`` [n_feasible, 2] in the order of the feasible candidates; ``single(g)`` one
    roll-out of g's stand-in chunk; ``feasible_aim(g)`` whether that chunk is feasible; ``known``
    the frozen path's (g, plate) pairs. Returns {variant: {"g", "converged", ...}} and the grid's
    argmin under "_grid"."""
    p_hat = np.asarray(p_hat, np.float64).reshape(2)
    h = np.asarray(h, np.float64).reshape(2)
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    index = np.flatnonzero(np.asarray(feasible, bool))
    grid_plates = np.asarray(grid_plates, np.float64).reshape(-1, 2)
    scores = np.linalg.norm(grid_plates - targets[index], axis=1)
    j = int(np.argmin(scores))
    best = int(index[j])
    g0 = targets[best].copy()
    kw = {"tolerance_m": tolerance_m, "p_hat": p_hat, "h": h, "a_lo": a_lo}
    out = {}

    def norms(ev):
        return [100.0 * float(np.linalg.norm(r)) for _g, r in ev]

    g, conv, ev10, stop, n = iterate(g0, single, feasible_aim, alpha=1.0, cap=lm.REFINE_MAX,
                                     known=known, **kw)  # fmt: skip
    out["frozen"] = {"g": g, "converged": conv, "stopped_infeasible": stop, "rollouts": n,
                     "residuals_cm": norms(ev10)}  # fmt: skip
    g, conv, ev30, stop, n = iterate(g0, single, feasible_aim, alpha=1.0, cap=LONG_CAP,
                                     known=known, **kw)  # fmt: skip
    out["cap30"] = {"g": g, "converged": conv, "stopped_infeasible": stop, "rollouts": n,
                    "iterations": len(ev30), "residuals_cm": norms(ev30)}  # fmt: skip
    g, conv, evd, stop, n = iterate(g0, single, feasible_aim, alpha=DAMPING, cap=DAMPED_CAP,
                                    known=known[:1], **kw)  # fmt: skip
    out["damped"] = {"g": g, "converged": conv, "stopped_infeasible": stop, "rollouts": n,
                     "residuals_cm": norms(evd)}  # fmt: skip
    if out["frozen"]["converged"] or out["frozen"]["stopped_infeasible"]:
        out["bestres"] = {"g": out["frozen"]["g"], "converged": out["frozen"]["converged"],
                          "rollouts": 0, "index": None}  # fmt: skip
    else:
        g, res, i = best_residual(ev10)
        out["bestres"] = {"g": g, "converged": False, "rollouts": 0, "index": i,
                          "residual_cm": 100.0 * res}  # fmt: skip
    for name, mask in (
        ("affine_local", local_mask(grid, best)[index]),
        ("affine_global", np.ones(len(index), bool)),
    ):
        g, jac = affine_fixed_point(targets[index][mask], grid_plates[mask], p_hat=p_hat, h=h,
                                    a_lo=a_lo)  # fmt: skip
        entry = {"points": int(mask.sum()), "jacobian": None if jac is None else jac.tolist(),
                 "fallback_grid_argmin": False, "converged": None}  # fmt: skip
        if g is None or not feasible_aim(g):
            g, entry["fallback_grid_argmin"] = g0.copy(), True
        plate = np.asarray(single(g), np.float64)
        entry |= {
            "g": g,
            "rollouts": 1,
            "residual_cm": 100.0 * float(np.linalg.norm(_residual(plate, g, p_hat, h, a_lo))),
        }
        out[name] = entry
    out["_grid"] = {"best": best, "g0": g0, "score_cm": 100.0 * float(scores[j])}
    return out


# ----- the development worker ---------------------------------------------------------------------
_STATE: dict = {"variant": "frozen"}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _wrapped_choose(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of, *, tolerance_m,
                    seed):  # fmt: skip
    """TASK-080's ``choose`` for W with every variant computed and logged; the aim returned is the
    task's variant's. The frozen variant must equal ``choose_from_grid``'s aim and convergence."""
    from embodied_jepa import lewm_c1m_v2_runtime as wrt

    if arm != "W":
        return _STATE["original_choose"](arm, p_hat, h, targets, chunks, feasible, predict,
                                         chunk_of, tolerance_m=tolerance_m, seed=seed)  # fmt: skip
    grid_out, singles, chunk_cache, plate_cache = {}, [], {}, {}

    def recording(commands):
        out = np.asarray(predict(commands), np.float64)
        if "grid" not in grid_out:
            grid_out["grid"] = out.copy()
        else:
            plate = out.reshape(2).copy()
            singles.append(plate)
            plate_cache[np.ascontiguousarray(commands[0], np.float32).tobytes()] = plate
        return out

    g_frozen, log = wrt.choose_from_grid(
        arm, p_hat, h, targets, chunks, feasible, recording, chunk_of,
        tolerance_m=tolerance_m, seed=seed,
    )  # fmt: skip
    if log.get("fallback_all_infeasible"):
        log["variants"] = {"note": "all candidates infeasible: every variant aims at p-hat"}
        return g_frozen, log

    def chunk(g):
        key = np.asarray(g, np.float64).tobytes()
        if key not in chunk_cache:
            chunk_cache[key] = chunk_of(np.asarray(g, np.float64))
        return chunk_cache[key]

    def feasible_aim(g):
        return bool(chunk(g)[1])

    def single(g):
        commands, ok = chunk(g)
        if not ok:
            raise RuntimeError("a roll-out of an infeasible aim")
        key = np.ascontiguousarray(commands, np.float32).tobytes()
        if key not in plate_cache:
            out = predict(np.asarray(commands)[None])
            plate_cache[key] = np.asarray(out, np.float64).reshape(2)
        return plate_cache[key]

    best = int(log["grid_index"])
    path = [np.asarray(targets[best], np.float64)] + [
        np.asarray(s["g"], np.float64) for s in log.get("iterations", []) if "g" in s
    ]
    known = list(zip(path, singles, strict=False))
    variants = all_variants(p_hat, h, targets, feasible, grid_out["grid"], single, feasible_aim,
                            known=known, tolerance_m=tolerance_m, a_lo=lm.A_LO)  # fmt: skip
    if not np.array_equal(variants["frozen"]["g"], np.asarray(g_frozen, np.float64)):
        raise RuntimeError("the frozen variant differs from choose_from_grid")
    if bool(variants["frozen"]["converged"]) != bool(log["converged"]):
        raise RuntimeError("the frozen variant's convergence differs from choose_from_grid")
    chosen = _STATE["variant"]
    log["variants"] = _plain(variants)
    log["executed_variant"] = chosen
    return np.asarray(variants[chosen]["g"], np.float64).copy(), log


def worker_init(config: dict) -> None:
    from embodied_jepa import lewm_pr_v2_runtime as prt

    prt.worker_init(config)
    if "original_choose" not in _STATE:
        _STATE["original_choose"] = prt.choose
    prt.choose = _wrapped_choose


def run_task(task: dict) -> dict:
    from embodied_jepa import lewm_pr_v2_runtime as prt

    if task.get("kind") == "attempt":
        variant = task.get("variant", "frozen")
        if variant not in EXECUTABLE:
            raise ValueError(f"not an executable variant: {variant}")
        _STATE["variant"] = variant
    out = prt.run_task(task)
    if isinstance(out, dict):
        out.pop("frame405", None)
        out.pop("corpus", None)
    return out


# ----- G-NI's power at a cohort size (design note §5) --------------------------------------------
POWER_SALT = 8301  # the design note's development salt (8301-8312 proposed for TASK-081)
COUPLINGS = ("overlap", "half", "independent")


def discordance(p_w: float, p_c: float, coupling: str) -> tuple[float, float]:
    """(P(W succeeds and C fails), P(W fails and C succeeds)) for the marginal rates given:
    "overlap" makes the outcomes as nested as the rates allow, "independent" independent, and
    "half" is halfway between in both discordant probabilities."""
    over = (max(p_w - p_c, 0.0), max(p_c - p_w, 0.0))
    ind = (p_w * (1.0 - p_c), (1.0 - p_w) * p_c)
    if coupling == "overlap":
        return over
    if coupling == "independent":
        return ind
    if coupling == "half":
        return ((over[0] + ind[0]) / 2.0, (over[1] + ind[1]) / 2.0)
    raise ValueError(coupling)


def ni_pass_table(n: int, margin: int, *, resamples: int = 4000, k_max: int = 40) -> np.ndarray:
    """pass[k+, k-]: whether TASK-080's G-NI estimator (the 2.5th percentile of the reset-bootstrap
    sum of W - C, ``lewm_pr_v2.paired_interval``'s form) is > -margin, for k+ resets won by W only
    and k- by C only out of n. The bootstrap sum depends on the resets only through (k+, k-)."""
    rng = np.random.default_rng(np.random.SeedSequence([POWER_SALT, int(n), int(margin)]))
    k_max = min(int(k_max), int(n))
    table = np.zeros((k_max + 1, k_max + 1), bool)
    for kp in range(k_max + 1):
        for km in range(k_max + 1):
            if kp + km > n:
                continue
            draws = rng.multinomial(n, [kp / n, km / n, (n - kp - km) / n], size=resamples)
            lo = np.percentile(draws[:, 0] - draws[:, 1], 2.5)
            table[kp, km] = lo > -margin
    return table


def ni_power(p_w: float, p_c: float, coupling: str, *, n: int, margin: int, table=None) -> float:
    """P(G-NI passes) over the multinomial of (k+, k-) at the discordance of ``coupling``."""
    from math import lgamma, log

    table = ni_pass_table(n, margin) if table is None else table
    q_p, q_m = discordance(p_w, p_c, coupling)
    q_0 = 1.0 - q_p - q_m
    total = 0.0
    k_max = table.shape[0] - 1
    for kp in range(k_max + 1):
        for km in range(k_max + 1):
            if kp + km > n or not table[kp, km]:
                continue
            if (q_p == 0 and kp) or (q_m == 0 and km) or (q_0 == 0 and kp + km < n):
                continue
            logp = lgamma(n + 1) - lgamma(kp + 1) - lgamma(km + 1) - lgamma(n - kp - km + 1)
            logp += (kp * log(q_p) if kp else 0.0) + (km * log(q_m) if km else 0.0)
            logp += (n - kp - km) * log(q_0) if n - kp - km else 0.0
            total += float(np.exp(logp))
    return total
