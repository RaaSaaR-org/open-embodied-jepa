"""TASK-075 simulation workers: a tau-calibration attempt and one multi-view re-render of a sealed
``apple-far-shift-v2`` root.

Protocol ``docs/experiments/apple_obs_ceiling_v2.md``. The worker is TASK-074's
(``lewm_planner_v2_runtime.worker_init``, itself TASK-073's): the v2 robot, P-3's private
kinematics, the bounds, P-3 and the perturber. On top of it this module builds, on first use, the
extra renderers of the new views (onboard 224, overview 224, the hand crop) and a 112 px renderer
for the reference view's hidden renders.

- **tau** (``run_planted_task``): K1's H-handover attempt, run through TASK-074's own
  ``run_attempt_task``, with the harness's plate schedule displaced by the planted error. Only
  the target that e9's place primitive is aimed at moves; the plate, the shift and P-3's pick
  are untouched. Privileged (it reads the true moved plate), a calibration, not a learned arm.
- **views** (``run_views_task``): TASK-073's e9 collector loop (``wm_critic_v2_runtime.
  run_collect_task``), re-run from the sealed root's own metadata, with the extra views captured
  at the rendered steps. The worker then checks the re-simulation against the sealed episode
  (every array exactly, the 112 px frames by the render rule) and writes the views file.

NumPy at import; MuJoCo and torch are imported by the worker initializer.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import obs_ceiling_v2 as oc
from embodied_jepa import plate_shift as ps
from embodied_jepa import wm_critic_v2_runtime as rtm

_V: dict = {}
COMPARED_ARRAYS = (
    "states",
    "base",
    "requested",
    "applied",
    "phase",
    "apple",
    "plate",
    "dropped",
    "hand_contact",
    "latched",
)
RENDER_SET = frozenset(oc.RENDER_STEPS)
HIDDEN_SET = frozenset(oc.HIDDEN_STEPS)


def worker_init(config: dict) -> None:
    lrt.worker_init(config)
    _V.clear()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----- tau: K1's H-handover with a planted target error -----------------------------------------
def planted_schedule(base, planted_m):
    """``(reset, shift) -> (step -> plate xy + planted)``: the harness schedule, displaced."""
    offset = np.asarray(planted_m, np.float64).reshape(2)

    def make(reset, shift):
        inner = base(reset, shift)
        return lambda step: np.asarray(inner(step), np.float64).reshape(2) + offset

    return make


def run_planted_task(task: dict) -> dict:
    original = lrt.plate_schedule
    lrt.plate_schedule = planted_schedule(original, task["planted_m"])
    try:
        out = lrt.run_attempt_task(task | {"kind": "attempt", "arm": "H-handover"})
    finally:
        lrt.plate_schedule = original
    out = {k: v for k, v in out.items() if k not in ("latents",)}
    out["planted_m"] = [float(v) for v in task["planted_m"]]
    out["level_cm"] = float(task["level_cm"])
    return out


def planted_check(record: dict, reset: dict, shift: dict | None) -> dict:
    """G-planted: every decision that did not fall back aimed at the true plate plus the planted
    error (to 1e-9 m)."""
    plate_at = rtm.plate_schedule(reset, shift)
    planted = np.asarray(record["planted_m"], np.float64)
    worst, fallbacks = 0.0, 0
    for d in record.get("decisions") or []:
        if d.get("fallback"):
            fallbacks += 1
            continue
        want = np.asarray(plate_at(int(d["step"])), np.float64) + planted
        worst = max(worst, float(np.abs(np.asarray(d["target"]) - want).max()))
    if worst > 1e-9:
        raise lp.GuardError(f"G-planted: a decision aimed {worst} m away from plate + error")
    return {"max_abs_target_error_m": worst, "fallbacks": fallbacks}


# ----- views: the extra renderers ---------------------------------------------------------------
def _views() -> dict:
    if "r224" not in _V:
        from embodied_jepa.hand_crop import HandCrop

        robot = rtm._W["robot"]
        mj, model = robot.mj, robot.model
        _V["r224"] = mj.Renderer(model, height=224, width=224)
        _V["r112"] = mj.Renderer(model, height=112, width=112)
        _V["crop"] = HandCrop(robot, render_size=320, crop_size=112)
        names = {b: mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)}
        apple = [b for b, n in names.items() if n == "apple"]
        plate = [b for b, n in names.items() if n and ("plate" in n or "container" in n)]
        if len(apple) != 1 or not plate:
            raise lp.GuardError(f"G-scene: apple {apple} or plate {plate} bodies not found")
        _V["hidden"] = {
            "apple_hidden": [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in apple],
            "plate_hidden": [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in plate],
        }
        used = {int(model.geom_group[g]) for g in range(model.ngeom)}
        if oc.HIDDEN_GROUP in used:
            raise lp.GuardError("G-scene: a geom already uses the hidden group")
    return _V


def _render(renderer, camera: str, hide: str | None = None) -> np.ndarray:
    robot = rtm._W["robot"]
    mj, model = robot.mj, robot.model
    geoms = _V["hidden"][hide] if hide else []
    saved = [int(model.geom_group[g]) for g in geoms]
    try:
        option = None
        if hide:
            for g in geoms:
                model.geom_group[g] = oc.HIDDEN_GROUP
            option = mj.MjvOption()
            option.geomgroup[oc.HIDDEN_GROUP] = 0
        if option is None:
            renderer.update_scene(robot.sim.data, camera=camera)
        else:
            renderer.update_scene(robot.sim.data, camera=camera, scene_option=option)
        return renderer.render().copy()
    finally:
        for g, group in zip(geoms, saved, strict=True):
            model.geom_group[g] = group


def capture(view: str, hide: str | None = None):
    """One frame of ``view`` at the current simulation state (uint8), and the crop window."""
    v = _views()
    camera = oc.VIEW_SPECS[view]["camera"]
    if view == "onboard112":
        return _render(v["r112"], camera, hide), None
    if view in ("onboard224", "overview224"):
        return _render(v["r224"], camera, hide), None
    if view == "handcrop":
        crop = v["crop"]
        image = _render(crop.renderer, camera, hide)
        top, left = crop.window()
        size = crop.crop_size
        return image[top : top + size, left : left + size].copy(), (top, left)
    raise lp.GuardError(f"unknown view {view!r}")


def warm_up() -> None:
    """RENDER_RULE['warm_up']: every extra renderer renders once, discarded, per root."""
    for view in oc.VIEWS:
        capture(view)


# ----- views: one sealed root re-simulated --------------------------------------------------------
def frame_check(ours: np.ndarray, stored: np.ndarray) -> dict:
    identical = within = outside = 0
    worst = {"max_level": 0, "pixels": 0}
    for a, b in zip(ours, stored, strict=True):
        if np.array_equal(a, b):
            identical += 1
            continue
        d = lp.wc.frame_difference(a, b)
        worst = {k: max(worst[k], d[k]) for k in worst}
        if lp.wc.render_difference_allowed(d):
            within += 1
        else:
            outside += 1
    return {"identical": identical, "within_rule": within, "outside_rule": outside, **worst}


def run_views_task(task: dict) -> dict:
    """``task``: ``episode_id``, ``meta`` (the sealed root's metadata), ``source_npz`` and its
    ``source_npz_sha256``, ``folder``. Returns the summary; writes ``<folder>/<id>.npz``."""
    from embodied_jepa.at_rest import AppleAtRestCheck

    started = time.monotonic()
    meta = task["meta"]
    robot = rtm._W["robot"]
    lower, upper = rtm._W["bounds"]
    _views()
    truth, scorer, counter, _obs, _facts = rt2.reset_and_look(robot, meta["seed"], meta["reset"])
    counter.remove()  # the privileged scripted collector, as sealed: truth is read by design
    warm_up()
    shift = meta.get("shift")
    offset = np.asarray(shift["vector"], np.float64) if shift else np.zeros(2)
    aim = offset + np.asarray(meta["misaim_m"], np.float64)
    expert = rt2.make_expert(rt2.perturbed_truth(truth, [0.0, 0.0], aim))
    perturber = rtm._W["perturber"](meta["noise_level"], meta["noise_seed"])
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"]) if shift else None
    check = AppleAtRestCheck(robot)
    rows = {k: [] for k in COMPARED_ARRAYS}
    ref, steps, hidden_steps, windows, windows_hidden = [], [], [], [], []
    frames = {v: [] for v in oc.RENDERED_VIEWS}
    hidden = {(v, k): [] for v in oc.VIEWS for k in oc.HIDDEN_KINDS}
    counter_rows = [0]

    def observe_row(obs):
        k = counter_rows[0]
        now = robot.sim.task_truth()
        rows["states"].append(rt2.state_of(obs))
        rows["apple"].append(np.asarray(now["object_position"], np.float32))
        rows["plate"].append(np.asarray(now["plate_position"], np.float32))
        rows["dropped"].append(bool(now["dropped"]))
        rows["hand_contact"].append(bool(now["hand_contact"]))
        if k in RENDER_SET:
            steps.append(k)
            ref.append(np.asarray(obs.images[fp2.CAMERA][0], np.uint8).copy())
            for view in oc.RENDERED_VIEWS:
                image, window = capture(view)
                frames[view].append(image)
                if window is not None:
                    windows.append(window)
        if k in HIDDEN_SET:
            hidden_steps.append(k)
            for view in oc.VIEWS:
                for kind in oc.HIDDEN_KINDS:
                    image, window = capture(view, kind)
                    hidden[(view, kind)].append(image)
                    if window is not None and kind == oc.HIDDEN_KINDS[0]:
                        windows_hidden.append(window)
        counter_rows[0] += 1

    def step(base, requested, phase):
        try:
            projection = robot.project_candidates(requested[None, None, None])
        except rt2.ContractError as error:
            if str(error) not in rt2.GUARD_REFUSALS:
                raise
            return "guard_refusal"
        if not bool(projection.feasible[0, 0]):
            return "infeasible_command"
        result = robot.execute(projection.actions[0, 0, 0])
        if result.applied_action is None:
            return result.reason or result.status or "rejected"
        if phase < fp2.SETTLE_PHASE:
            expert.advance(result)
        score = scorer.evaluate()
        check.record()
        rows["base"].append(np.asarray(base, np.float32))
        rows["requested"].append(requested)
        rows["applied"].append(np.asarray(result.applied_action, np.float32))
        rows["phase"].append(phase)
        rows["latched"].append(bool(score["success"]))
        observe_row(robot.observe())
        return None

    try:
        observe_row(robot.observe())
        termination, n = "step_limit", 0
        while not expert.done and n < fp2.MAX_POLICY_STEPS:
            phase = expert.phase_index
            base = np.asarray(expert.action(robot), np.float32)
            requested, _kind = perturber(base)
            requested = np.clip(requested, lower, upper).astype(np.float32)
            stop = step(base, requested, phase)
            if stop:
                termination = stop
                break
            n += 1
        else:
            termination = "policy_complete" if expert.done else "step_limit"
        complete = termination in ("policy_complete", "step_limit")
        if complete:
            settle = rt2.settle_command(rtm._W["bounds"])
            for _ in range(fp2.SETTLE_STEPS):
                stop = step(settle, settle, fp2.SETTLE_PHASE)
                if stop:
                    termination, complete = f"settle_{stop}", False
                    break
    finally:
        if hook is not None:
            hook.remove()
    robot.stop(termination)
    arrays = {
        "states": np.asarray(rows["states"], np.float64),
        "base": np.asarray(rows["base"], np.float32).reshape(-1, 14),
        "requested": np.asarray(rows["requested"], np.float32).reshape(-1, 14),
        "applied": np.asarray(rows["applied"], np.float32).reshape(-1, 14),
        "phase": np.asarray(rows["phase"], np.int8),
        "apple": np.asarray(rows["apple"], np.float32),
        "plate": np.asarray(rows["plate"], np.float32),
        "dropped": np.asarray(rows["dropped"], bool),
        "hand_contact": np.asarray(rows["hand_contact"], bool),
        "latched": np.asarray(rows["latched"], bool),
    }
    if sha256_file(task["source_npz"]) != task["source_npz_sha256"]:
        raise lp.GuardError(f"G-data: {task['episode_id']} differs from its sealed hash")
    with np.load(task["source_npz"]) as stored:
        equal = {
            k: bool(
                stored[k].shape == arrays[k].shape
                and np.array_equal(stored[k], arrays[k].astype(stored[k].dtype))
            )
            for k in COMPARED_ARRAYS
        }
        checked = frame_check(
            np.asarray(ref, np.uint8), stored["frames"][np.asarray(steps, np.int64)]
        )
    out_arrays = {
        "steps": np.asarray(steps, np.int32),
        "hidden_steps": np.asarray(hidden_steps, np.int32),
        "crop_window": np.asarray(windows, np.int16).reshape(-1, 2),
        "crop_window_hidden": np.asarray(windows_hidden, np.int16).reshape(-1, 2),
    }
    for view in oc.RENDERED_VIEWS:
        out_arrays[view] = np.asarray(frames[view], np.uint8)
    for (view, kind), images in hidden.items():
        out_arrays[f"{view}__{kind}"] = np.asarray(images, np.uint8)
    path = Path(task["folder"]) / f"{task['episode_id']}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez_compressed(path, **out_arrays)
    return {
        "episode_id": task["episode_id"],
        "rows": int(len(arrays["phase"])),
        "termination": termination,
        "rendered_steps": len(steps),
        "hidden_steps": len(hidden_steps),
        "arrays_equal": equal,
        "frame_check": checked,
        "npz_sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "seconds": time.monotonic() - started,
    }


def run_task(task: dict) -> dict:
    kind = task["kind"]
    if kind == "frame":
        return rtm.run_frame_task(task)
    if kind == "planted":
        return run_planted_task(task)
    if kind == "views":
        return run_views_task(task)
    if kind == "collect":
        return lrt.run_collect_task(task)
    raise lp.GuardError(f"unknown task kind {kind!r}")
