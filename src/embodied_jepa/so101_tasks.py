"""SO-101 tasks ``so101_cube_lift_v0`` and ``so101_cube_place_v0`` (SO-101 S2, TASK-098).

Each task owns its reset distribution (declared below), its goal, its termination and its
scoring. Scoring reads simulator truth (``SO101Simulation.task_truth()``) only, and nothing here
enters a model input or a planning cost: ``goal()`` returns the declared goal for logs and for
scripted references, never an observation.

Both scorers are *held-window* scorers, never latched on a transient: a condition counts only
when it holds on every control step of a declared window, and the episode terminates on the step
that completes that window (or on a lost cube, or at ``max_steps``). The pure functions
``lift_outcome`` and ``place_outcome`` work on per-step truth arrays, so the same code scores a
live episode (``evaluate()`` once per control step) and a synthetic trajectory in the tests.

- **Lift** (``LIFT_V0``): the cube centre at least ``lift_height_m`` above its resting height,
  with a gripper contact, on every step of ``hold_steps`` consecutive control steps.
- **Place** (``PLACE_V0``): first a pick (the cube centre at least ``pick_height_m`` above
  rest with a gripper contact for ``pick_steps`` consecutive steps), then, on steps after the
  pick, the cube **at rest** inside the target square on every step of ``rest_steps``
  consecutive steps: centre inside the square, resting on the table (height), linear and angular
  speed below their bars, xy drift over the window below a bar, and no gripper contact.

Thresholds and reset regions are declared, with their rationale, in
``docs/experiments/so101_cube_tasks_dev.md``; they were fixed before the scripted follower's
measurement and are not tuned on its outcome. Development-only measurements, no claim.

NumPy only at import; the MuJoCo transport is ``so101_simulation.py``. Opt-in: nothing in
``import embodied_jepa`` loads this module; ``config.py`` registers it as a lazy string.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0

# The scene's constants (so101_simulation.py), repeated so this module needs no MuJoCo.
CUBE_HALF_SIZE_M = 0.015
TABLE_TOP_Z = 0.0
REST_Z = TABLE_TOP_Z + CUBE_HALF_SIZE_M
# The table's top extends over x in [-0.20, 0.50], y in [-0.40, 0.40] (so101_simulation.py).
TABLE_XY = ((-0.20, 0.50), (-0.40, 0.40))
LIFT_TASK = "so101_cube_lift_v0"
PLACE_TASK = "so101_cube_place_v0"
# Separate reset streams per task, so a seed's lift and place resets are unrelated draws.
_TASK_STREAM = {LIFT_TASK: 9801, PLACE_TASK: 9802}


def _positive(cfg, floats, ints):
    for name in floats:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ints:
        value = getattr(cfg, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ResetDistribution:
    """Declared reset distribution (world metres on the table top; yaw in radians about +z).

    Cube xy and target xy are uniform on their boxes; cube yaw is uniform on ``cube_yaw_rad``
    (±45° covers every face orientation of a cube). A place target is redrawn until its centre is
    at least ``min_cube_target_m`` from the cube, which (≥ 0.10 m > √2 × 0.055 m) keeps the
    starting cube wholly outside the target square.
    """

    cube_x_m: tuple[float, float] = (0.18, 0.23)
    cube_y_m: tuple[float, float] = (-0.10, 0.10)
    cube_yaw_rad: tuple[float, float] = (-np.pi / 4, np.pi / 4)
    target_x_m: tuple[float, float] = (0.17, 0.21)
    target_y_m: tuple[float, float] = (-0.12, 0.12)
    min_cube_target_m: float = 0.10
    max_draws: int = 1000

    def __post_init__(self):
        for name in ("cube_x_m", "cube_y_m", "cube_yaw_rad", "target_x_m", "target_y_m"):
            low, high = getattr(self, name)
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ValueError(f"{name} must be an ordered finite pair")
        if not np.isfinite(self.min_cube_target_m) or self.min_cube_target_m < 0:
            raise ValueError("min_cube_target_m must be finite and nonnegative")
        if type(self.max_draws) is not int or self.max_draws < 1:
            raise ValueError("max_draws must be a positive integer")


RESETS_V0 = ResetDistribution()


@dataclass(frozen=True)
class LiftThresholds:
    version: str = "so101_cube_lift_v0"
    lift_height_m: float = 0.05  # cube centre at least 5 cm above its resting height
    hold_steps: int = 20  # every step of 1.0 s at 20 Hz
    require_gripper_contact: bool = True
    max_steps: int = 200  # 10 s
    lost_below_z_m: float = -0.05  # cube centre below the table top: fell off the table

    def __post_init__(self):
        _positive(self, ("lift_height_m",), ("hold_steps", "max_steps"))
        if self.hold_steps > self.max_steps:
            raise ValueError("hold_steps must fit inside max_steps")
        if not np.isfinite(self.lost_below_z_m) or self.lost_below_z_m >= TABLE_TOP_Z:
            raise ValueError("lost_below_z_m must be finite and below the table top")


@dataclass(frozen=True)
class PlaceThresholds:
    version: str = "so101_cube_place_v0"
    target_half_size_m: float = 0.04  # the marker's 8 cm square (axis-aligned); centre inside
    pick_height_m: float = 0.02  # pick: centre 2 cm above rest with a gripper contact ...
    pick_steps: int = 5  # ... on 5 consecutive steps (0.25 s)
    rest_height_tolerance_m: float = 0.003  # centre within 3 mm of the resting height
    max_speed_m_s: float = 0.005
    max_angular_speed_rad_s: float = 0.1
    max_window_drift_m: float = 0.002  # xy range of the centre over the window
    rest_steps: int = 20  # every step of 1.0 s at 20 Hz, all after the pick
    max_steps: int = 320  # 16 s
    lost_below_z_m: float = -0.05

    def __post_init__(self):
        _positive(
            self,
            (
                "target_half_size_m",
                "pick_height_m",
                "rest_height_tolerance_m",
                "max_speed_m_s",
                "max_angular_speed_rad_s",
                "max_window_drift_m",
            ),
            ("pick_steps", "rest_steps", "max_steps"),
        )
        if self.pick_steps + self.rest_steps > self.max_steps:
            raise ValueError("pick_steps + rest_steps must fit inside max_steps")
        if not np.isfinite(self.lost_below_z_m) or self.lost_below_z_m >= TABLE_TOP_Z:
            raise ValueError("lost_below_z_m must be finite and below the table top")


LIFT_V0 = LiftThresholds()
PLACE_V0 = PlaceThresholds()


def sample_reset(task: str, seed: int, distribution: ResetDistribution = RESETS_V0) -> dict:
    """The declared reset for ``task`` and ``seed``: cube xy and yaw, plus target xy for place.

    Lift has no target; its scene still needs a marker, which is put at a fixed spot far from
    every lift cube (x = 0.30 m, y = -0.20 m) so a lift episode never shows a random target.
    """
    if task not in _TASK_STREAM:
        raise ValueError(f"unknown SO-101 task {task!r}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    d = distribution
    rng = np.random.default_rng([_TASK_STREAM[task], seed])
    cube = np.array([rng.uniform(*d.cube_x_m), rng.uniform(*d.cube_y_m)])
    yaw = float(rng.uniform(*d.cube_yaw_rad))
    if task == LIFT_TASK:
        target = np.array([0.30, -0.20])
    else:
        for _ in range(d.max_draws):
            target = np.array([rng.uniform(*d.target_x_m), rng.uniform(*d.target_y_m)])
            if np.linalg.norm(target - cube) >= d.min_cube_target_m:
                break
        else:
            raise ValueError("could not draw a target far enough from the cube")
    return {"cube_xy": cube.tolist(), "cube_yaw": yaw, "target_xy": target.tolist()}


def _runs(mask):
    """Length of the run of True values ending at each index."""
    out = np.zeros(len(mask), dtype=int)
    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        out[i] = count
    return out


def _lost(cube_position, lost_below_z_m):
    (x0, x1), (y0, y1) = TABLE_XY
    p = cube_position
    off_table = (p[:, 0] < x0) | (p[:, 0] > x1) | (p[:, 1] < y0) | (p[:, 1] > y1)
    return off_table | (p[:, 2] < lost_below_z_m)


def _first(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def _check_arrays(cube_position, gripper_contact, *others):
    position = np.asarray(cube_position, dtype=float)
    contact = np.asarray(gripper_contact, dtype=bool)
    steps = len(position)
    if position.ndim != 2 or position.shape[1] != 3 or contact.shape != (steps,):
        raise ValueError("expected cube positions [T, 3] and gripper contacts [T]")
    arrays = [np.asarray(a, dtype=float) for a in others]
    for array in arrays:
        if array.shape != position.shape:
            raise ValueError("every per-step vector array must be [T, 3]")
    if not all(np.isfinite(a).all() for a in (position, *arrays)):
        raise ValueError("truth arrays must be finite")
    return position, contact, arrays


def lift_outcome(cube_position, gripper_contact, thresholds: LiftThresholds = LIFT_V0) -> dict:
    """Score a lift trajectory of per-control-step truth.

    ``success_step`` is the first step that completes ``hold_steps`` consecutive steps with the
    cube centre at least ``lift_height_m`` above rest (and a gripper contact when required). A
    lost cube before that step is a failure, as is a hold that ends early (a drop).
    """
    cfg = thresholds
    p, contact, _ = _check_arrays(cube_position, gripper_contact)
    high = p[:, 2] - REST_Z >= cfg.lift_height_m
    held = high & contact if cfg.require_gripper_contact else high
    run = _runs(held)
    success_step = _first(run >= cfg.hold_steps)
    lost_step = _first(_lost(p, cfg.lost_below_z_m))
    if success_step is not None and lost_step is not None and lost_step <= success_step:
        success_step = None
    return {
        "version": cfg.version,
        "success": success_step is not None,
        "success_step": success_step,
        "lost_step": lost_step,
        "longest_hold_steps": int(run.max()) if len(run) else 0,
        "max_lift_m": float((p[:, 2] - REST_Z).max()) if len(p) else 0.0,
        "steps": len(p),
    }


def place_outcome(
    cube_position,
    cube_velocity,
    cube_angular_velocity,
    gripper_contact,
    target_xy,
    thresholds: PlaceThresholds = PLACE_V0,
) -> dict:
    """Score a place trajectory of per-control-step truth.

    ``pick_step`` is the first step that completes ``pick_steps`` consecutive steps with the
    cube at least ``pick_height_m`` above rest and a gripper contact. ``success_step`` is the
    first step that completes ``rest_steps`` consecutive steps, all after ``pick_step``, on which
    the cube is at rest inside the target square: centre within ``target_half_size_m`` of the
    target in x and y, height within ``rest_height_tolerance_m`` of rest, linear and angular
    speed at most their bars and no gripper contact; and the centre's xy range over those steps
    is at most ``max_window_drift_m``. A lost cube before ``success_step`` is a failure.
    """
    cfg = thresholds
    p, contact, (v, w) = _check_arrays(
        cube_position, gripper_contact, cube_velocity, cube_angular_velocity
    )
    target = np.asarray(target_xy, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target_xy must be a finite pair")
    steps = len(p)
    lifted = (p[:, 2] - REST_Z >= cfg.pick_height_m) & contact
    pick_step = _first(_runs(lifted) >= cfg.pick_steps)
    offset = np.abs(p[:, :2] - target)
    inside = np.all(offset <= cfg.target_half_size_m, axis=1)
    resting = np.abs(p[:, 2] - REST_Z) <= cfg.rest_height_tolerance_m
    still = (np.linalg.norm(v, axis=1) <= cfg.max_speed_m_s) & (
        np.linalg.norm(w, axis=1) <= cfg.max_angular_speed_rad_s
    )
    ok = inside & resting & still & ~contact
    if pick_step is not None:
        ok[: pick_step + 1] = False
    else:
        ok[:] = False
    run = _runs(ok)
    success_step = None
    for i in np.flatnonzero(run >= cfg.rest_steps):
        window = p[i - cfg.rest_steps + 1 : i + 1, :2]
        if np.all(window.max(axis=0) - window.min(axis=0) <= cfg.max_window_drift_m):
            success_step = int(i)
            break
    lost_step = _first(_lost(p, cfg.lost_below_z_m))
    if success_step is not None and lost_step is not None and lost_step <= success_step:
        success_step = None
    final = p[-1] if steps else np.full(3, np.nan)
    return {
        "version": cfg.version,
        "success": success_step is not None,
        "picked": pick_step is not None,
        "pick_step": pick_step,
        "success_step": success_step,
        "lost_step": lost_step,
        "longest_rest_steps": int(run.max()) if steps else 0,
        "final_offset_xy_m": (final[:2] - target).tolist(),
        "final_inside": bool(inside[-1]) if steps else False,
        "steps": steps,
    }


class _SO101CubeTask:
    """Shared reset, goal, per-step truth recording and termination."""

    name = ""

    def __init__(self, embodiment, thresholds=None, reset_distribution=None):
        if getattr(embodiment, "action_schema", None) is not JOINT_DELTA_SO101_V0:
            raise ValueError(f"{self.name} needs the SO-101 embodiment (joint_delta_so101_v0)")
        self.robot = embodiment
        self.thresholds = thresholds or self._default_thresholds()
        self.reset_distribution = reset_distribution or RESETS_V0
        self.reset_spec: dict | None = None
        self._clear()

    def _clear(self):
        self.rows = {k: [] for k in ("position", "velocity", "angular", "contact")}
        self.last_time = -1.0
        self.outcome: dict | None = None

    def sample_reset(self, seed: int) -> dict:
        return sample_reset(self.name, seed, self.reset_distribution)

    def reset(self, seed: int) -> dict:
        """Reset the scene to the declared reset for ``seed``; returns the goal."""
        self.reset_spec = self.sample_reset(seed)
        self.robot.reset(seed, **self.reset_spec)
        self._clear()
        return self.goal()

    def start(self, reset_spec: dict) -> None:
        """Start scoring an episode whose scene was reset to ``reset_spec`` by the caller."""
        self.reset_spec = dict(reset_spec)
        self._clear()

    def evaluate(self) -> dict:
        """Record this control step's truth and return the score so far (scoring only)."""
        if self.reset_spec is None:
            raise ValueError("reset() or start() the task before evaluating")
        truth = self.robot.sim.task_truth()
        now = float(truth["timestamp"])
        if now <= self.last_time:
            raise ValueError("evaluate once per control step, after resetting the task")
        self.last_time = now
        self.rows["position"].append(np.asarray(truth["cube_position"], float).copy())
        self.rows["velocity"].append(np.asarray(truth["cube_velocity"], float).copy())
        self.rows["angular"].append(np.asarray(truth["cube_angular_velocity"], float).copy())
        self.rows["contact"].append(bool(truth["gripper_contact"]))
        self.outcome = self._score()
        return self.status()

    def status(self) -> dict:
        if self.outcome is None:
            raise ValueError("no step was evaluated")
        out = self.outcome
        steps = out["steps"]
        if out["success"]:
            reason = "success"
        elif out["lost_step"] is not None:
            reason = "cube_lost"
        elif steps >= self.thresholds.max_steps:
            reason = "timeout"
        else:
            reason = None
        return {**out, "done": reason is not None, "termination_reason": reason}

    @property
    def done(self) -> bool:
        return self.outcome is not None and self.status()["done"]


class SO101CubeLiftTask(_SO101CubeTask):
    """``so101_cube_lift_v0``: lift the cube and hold it (see ``lift_outcome``)."""

    name = LIFT_TASK

    @staticmethod
    def _default_thresholds():
        return LIFT_V0

    def goal(self) -> dict:
        return {
            "task": self.name,
            "lift_height_m": self.thresholds.lift_height_m,
            "hold_steps": self.thresholds.hold_steps,
            "thresholds": asdict(self.thresholds),
        }

    def _score(self):
        return lift_outcome(
            np.asarray(self.rows["position"]), np.asarray(self.rows["contact"]), self.thresholds
        )


class SO101CubePlaceTask(_SO101CubeTask):
    """``so101_cube_place_v0``: pick the cube and leave it at rest in the target square."""

    name = PLACE_TASK

    @staticmethod
    def _default_thresholds():
        return PLACE_V0

    def goal(self) -> dict:
        if self.reset_spec is None:
            raise ValueError("reset() the task first")
        return {
            "task": self.name,
            "target_xy": list(self.reset_spec["target_xy"]),
            "target_half_size_m": self.thresholds.target_half_size_m,
            "thresholds": asdict(self.thresholds),
        }

    def _score(self):
        return place_outcome(
            np.asarray(self.rows["position"]),
            np.asarray(self.rows["velocity"]),
            np.asarray(self.rows["angular"]),
            np.asarray(self.rows["contact"]),
            self.reset_spec["target_xy"],
            self.thresholds,
        )
