"""GR00T reference baseline: the GR00T E2E tutorial's sim evaluation, run as-is (development).

This is NVIDIA's fine-tuned GR00T N1.7 policy, served by the owner's already-running GR00T
server, evaluated in the tutorial's Arena scene. It is GR00T's result, not a LeWM or learned
project result. ``arena_policy_runner.py`` deliberately refuses remote and GR00T policies and
stays that way; this separate script allows exactly one command, the tutorial's
(``docs/ARENA.md``, "GR00T reference baseline"):

    policy_runner.py --headless --policy_type <POLICY_TYPE>
      --policy_config_yaml_path <POLICY_CONFIG>
      --remote_host localhost --remote_port 5555 (--num_steps N | --num_episodes N) [--num_envs N]
      --enable_cameras galileo_g1_static_pick_and_place --object apple_01_objaverse_robolab
      --destination clay_plates_hot3d_robolab --embodiment g1_wbc_agile_joint

The runner argv is built here, never passed through: host and port are pinned, and any kill,
exit, host, port or policy flag on this script's own command line is refused. The GR00T client
is guarded in-process: only the ``ping``, ``get_action``, ``reset`` and ``get_modality_config``
endpoints may be called (never ``kill``), ``shutdown_remote`` is a no-op, and the socket gets
send/receive timeouts so an unreachable or busy server fails the run instead of hanging it.

Modes:
- ``tutorial``: the runner unchanged. Arena's metrics are the tutorial's.
- ``settle``: the same, except Arena's ``success`` term is recorded every step but does not end
  the episode, so every episode runs to Arena's own 6 s time-out with GR00T acting. Used for the
  stricter ``apple_at_rest_arena_v0`` verdict; Arena's printed ``success_rate`` is then 0 by
  construction, and the Arena-rule verdict is the term's first firing (identical trajectory up
  to that step).

Both modes record, evaluator-only (never a policy input), per step and env: apple and plate
pose and velocity, the apple-plate and total contact forces, the nearest Dex3 link distance,
every termination term, and GR00T ``get_action`` latencies; plus a few head-camera frames.

Container-only, via ``scripts/isaac/run_isaac.sh arena_gr00t_baseline.py outputs/<new> --mode
tutorial --num_episodes 30``. The pure functions above ``main`` are host-testable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

RUNNER = "/workspaces/isaaclab_arena/isaaclab_arena/evaluation/policy_runner.py"
ARENA_ROOT = "/workspaces/isaaclab_arena"
POLICY_TYPE = (
    "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy"
)
POLICY_CONFIG = "isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml"
HOST = "localhost"
PORT = 5555
SCENE_ARGS = (
    "galileo_g1_static_pick_and_place",
    "--object",
    "apple_01_objaverse_robolab",
    "--destination",
    "clay_plates_hot3d_robolab",
    "--embodiment",
    "g1_wbc_agile_joint",
)
ALLOWED_ENDPOINTS = frozenset({"ping", "get_action", "reset", "get_modality_config"})
CLIENT_TIMEOUT_MS = 60_000
# Flags this script's own command line must never carry (the runner argv is built here).
FORBIDDEN_TOKENS = ("kill", "exit", "remote", "host", "port", "policy", "shutdown")


# ------------------------------------------------------------------ the strict rule (declared)
WINDOW_STEPS = 50  # the final 1.0 s at 50 Hz
PLATE_RADIUS_M = 0.04
SUPPORT_FORCE_N = 0.5
MAX_SPEED_M_S = 0.001
MAX_OTHER_CONTACT_N = 0.1
STRICT_VERSION = "apple_at_rest_arena_v0"


def parse_args(argv: list[str]) -> argparse.Namespace:
    """This script's own CLI. Refuses kill/exit/host/port/policy flags and anything unknown."""
    for a in argv:
        low = a.lower()
        if low.startswith("-") and any(t in low for t in FORBIDDEN_TOKENS):
            sys.exit(f"refusing {a!r}: the tutorial command is built here (host/port pinned)")
    p = argparse.ArgumentParser(prog="arena_gr00t_baseline.py", allow_abbrev=False)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("tutorial", "settle"), default="tutorial")
    length = p.add_mutually_exclusive_group(required=True)
    length.add_argument("--num_steps", type=int)
    length.add_argument("--num_episodes", type=int)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--frame_episodes", type=int, default=3, help="save frames for the first N")
    args = p.parse_args(argv)  # unknown flags -> SystemExit
    for name in ("num_steps", "num_episodes", "num_envs"):
        v = getattr(args, name)
        if v is not None and v < 1:
            sys.exit(f"--{name} must be >= 1")
    return args


def runner_argv(args: argparse.Namespace) -> list[str]:
    """The tutorial's evaluation command (headless instead of ``--viz kit``)."""
    length = (
        ["--num_steps", str(args.num_steps)]
        if args.num_steps is not None
        else ["--num_episodes", str(args.num_episodes)]
    )
    argv = [
        "--headless",
        "--policy_type",
        POLICY_TYPE,
        "--policy_config_yaml_path",
        POLICY_CONFIG,
        "--remote_host",
        HOST,
        "--remote_port",
        str(PORT),
        *length,
    ]
    if args.num_envs != 1:
        argv += ["--num_envs", str(args.num_envs)]
    argv += ["--enable_cameras", *SCENE_ARGS]
    check_runner_argv(argv)
    return argv


def check_runner_argv(argv: list[str]) -> None:
    """Last check on what reaches the runner: pinned host/port, this policy, no kill flag."""
    low = [a.lower() for a in argv]
    if any("kill" in a or "exit" in a or "shutdown" in a for a in low):
        raise SystemExit("refusing a kill/exit flag")
    if argv.count("--remote_host") != 1 or argv[argv.index("--remote_host") + 1] != HOST:
        raise SystemExit("remote host must be pinned to localhost")
    if argv.count("--remote_port") != 1 or argv[argv.index("--remote_port") + 1] != str(PORT):
        raise SystemExit("remote port must be pinned to 5555")
    if argv.count("--policy_type") != 1 or argv[argv.index("--policy_type") + 1] != POLICY_TYPE:
        raise SystemExit("only the tutorial's GR00T remote policy is allowed")
    if any(a.startswith(("--remote_", "--policy_type=")) and "=" in a for a in argv):
        raise SystemExit("no --flag=value forms for remote/policy flags")


def guard_client(client_cls, latencies: list | None = None, timeout_ms: int = CLIENT_TIMEOUT_MS):
    """Patch a GR00T ``PolicyClient`` class in place: endpoint allow-list, no kill, timeouts."""
    if getattr(client_cls, "_oej_guarded", False):
        return client_cls
    call = client_cls.call_endpoint
    init_socket = getattr(client_cls, "_init_socket", None)

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        if endpoint not in ALLOWED_ENDPOINTS:
            raise PermissionError(f"GR00T endpoint {endpoint!r} is not allowed (client only)")
        t0 = time.perf_counter()
        out = call(self, endpoint, data, requires_input)
        if latencies is not None and endpoint == "get_action":
            latencies.append(time.perf_counter() - t0)
        return out

    def kill_server(self):
        raise PermissionError("kill_server is not allowed: the GR00T server is the owner's")

    client_cls.call_endpoint = call_endpoint
    client_cls.kill_server = kill_server
    if init_socket is not None:

        def _init_socket(self):
            init_socket(self)
            import zmq

            self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)

        client_cls._init_socket = _init_socket
    client_cls._oej_guarded = True
    return client_cls


def strict_verdict(rows: dict[str, np.ndarray], terminated_by: str) -> dict:
    """``apple_at_rest_arena_v0`` on one episode's per-step truth (declared before any run).

    ``rows``: ``apple_pos``/``plate_pos``/``apple_vel`` [T, 3], ``plate_force``/
    ``other_force`` [T]. Every one of the final ``WINDOW_STEPS`` steps must be inside 4 cm of
    the plate centre (xy), supported by the plate (>= 0.5 N), still (<= 0.001 m/s) and released
    (non-plate contact <= 0.1 N). An episode not ended by ``time_out`` fails.
    """
    apple = np.asarray(rows["apple_pos"], float)
    plate = np.asarray(rows["plate_pos"], float)
    vel = np.asarray(rows["apple_vel"], float)
    pf = np.asarray(rows["plate_force"], float)
    of = np.asarray(rows["other_force"], float)
    steps = len(apple)
    out = {"version": STRICT_VERSION, "terminated_by": terminated_by, "steps": steps}
    if terminated_by != "time_out" or steps < WINDOW_STEPS:
        return {**out, "at_rest": False, "reason": "episode did not run to time_out"}
    w = slice(steps - WINDOW_STEPS, steps)
    dist = np.linalg.norm(apple[w, :2] - plate[w, :2], axis=1)
    speed = np.linalg.norm(vel[w], axis=1)
    inside, supported = dist <= PLATE_RADIUS_M, pf[w] >= SUPPORT_FORCE_N
    still, released = speed <= MAX_SPEED_M_S, of[w] <= MAX_OTHER_CONTACT_N
    return {
        **out,
        "at_rest": bool(np.all(inside & supported & still & released)),
        "inside_all": bool(inside.all()),
        "supported_all": bool(supported.all()),
        "still_all": bool(still.all()),
        "released_all": bool(released.all()),
        "final_distance_m": float(dist[-1]),
        "max_distance_m": float(dist.max()),
        "max_speed_m_s": float(speed.max()),
        "min_plate_force_n": float(pf[w].min()),
        "max_other_force_n": float(of[w].max()),
    }


# ------------------------------------------------------------------ container-only part
class _Recorder:
    """Per-step, per-env truth (pre-reset, captured inside the success term) and episodes."""

    def __init__(self, out_dir, mode: str, frame_episodes: int):
        from pathlib import Path

        self.out = Path(out_dir)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        (self.out / "traces").mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.frame_episodes = frame_episodes
        self.snapshot = None
        self.rows = None
        self.episode_index = None
        self.next_episode = 0
        self.ep_file = open(self.out / "episodes.jsonl", "w")  # noqa: SIM115 - closed in close()
        self.latencies: list[float] = []
        self.written = 0
        self.t0 = time.time()

    # called inside the success term, before any auto-reset
    def capture(self, env, value, params):
        import torch
        import warp as wp

        def t(x):
            return (x if isinstance(x, torch.Tensor) else wp.to_torch(x)).detach().cpu().numpy()

        u = env.unwrapped
        scene = u.scene
        origin = t(scene.env_origins)
        apple = scene[params["object_cfg"].name]
        sensor = scene[params["contact_sensor_cfg"].name]
        plate = scene[u.cfg.isaaclab_arena_env.task.destination_location.name]
        robot = scene["robot"]
        n = u.num_envs
        fm = t(sensor.data.force_matrix_w).reshape(n, -1, 3).sum(axis=1)
        net = t(sensor.data.net_forces_w).reshape(n, -1, 3).sum(axis=1)
        ap = t(apple.data.root_pos_w) - origin
        names = list(robot.data.body_names)
        hand_idx = [i for i, b in enumerate(names) if "hand" in b]
        bodies = t(robot.data.body_pos_w)[:, hand_idx] - origin[:, None]
        self.snapshot = {
            "success_raw": t(value).astype(bool).reshape(n),
            "apple_pos": ap,
            "apple_quat": t(apple.data.root_quat_w),
            "apple_vel": t(apple.data.root_lin_vel_w),
            "plate_pos": t(plate.data.root_pos_w) - origin,
            "plate_vel": t(plate.data.root_lin_vel_w),
            "plate_force": np.linalg.norm(fm, axis=-1),
            "net_force": np.linalg.norm(net, axis=-1),
            "other_force": np.linalg.norm(net - fm, axis=-1),
            "hand_dist": np.linalg.norm(bodies - ap[:, None], axis=-1).min(axis=1),
            "pelvis_pos": t(robot.data.root_pos_w) - origin,
        }

    def _new_episode(self, i):
        self.episode_index[i] = self.next_episode
        self.next_episode += 1
        self.rows[i] = {k: [] for k in self.snapshot} if self.snapshot else None
        self.ep_start[i] = time.time()

    def after_step(self, env, obs, prev_obs, terminated, truncated):
        u = env.unwrapped
        n = u.num_envs
        if self.rows is None:
            self.rows = [None] * n
            self.episode_index = [0] * n
            self.ep_start = [0.0] * n
            self.steps = [0] * n
            for i in range(n):
                self._new_episode(i)
        tm = u.termination_manager
        terms = {
            name: tm.get_term(name).detach().cpu().numpy().reshape(n) for name in tm.active_terms
        }
        term_mask = terminated.detach().cpu().numpy().reshape(n)
        trunc_mask = truncated.detach().cpu().numpy().reshape(n)
        for i in range(n):
            if self.rows[i] is None:
                self.rows[i] = {k: [] for k in self.snapshot}
            for k, v in self.snapshot.items():
                self.rows[i][k].append(np.asarray(v[i]))
            ep = self.episode_index[i]
            step = self.steps[i]
            if ep < self.frame_episodes and (step % 50 == 0):
                self._save_frame(prev_obs, i, f"ep{ep:03d}_step{step:03d}")
            self.steps[i] += 1
            if term_mask[i] or trunc_mask[i]:
                if ep < self.frame_episodes:
                    self._save_frame(prev_obs, i, f"ep{ep:03d}_step{step:03d}_last")
                fired = [name for name, v in terms.items() if v[i]]
                self._finish(i, fired, bool(trunc_mask[i]))
                self.steps[i] = 0
                self._new_episode(i)

    def _save_frame(self, obs, i, tag):
        try:
            rgb = obs["camera_obs"]["robot_head_cam_rgb"][i].detach().cpu().numpy()[..., :3]
            from PIL import Image

            Image.fromarray(rgb.astype(np.uint8)).save(self.out / "frames" / f"{tag}_env{i}.png")
        except Exception as exc:  # noqa: BLE001 - report-only
            print(f"[gr00t-baseline] frame {tag} not saved: {exc}", flush=True)

    def _finish(self, i, fired, truncated):
        rows = {k: np.stack(v) for k, v in self.rows[i].items()}
        ep = self.episode_index[i]
        np.savez_compressed(self.out / "traces" / f"ep{ep:03d}_env{i}.npz", **rows)
        succ = rows["success_raw"]
        first = int(np.argmax(succ)) if succ.any() else None
        if self.mode == "tutorial":
            ended = (
                "success"
                if "success" in fired
                else (
                    "time_out"
                    if "time_out" in fired
                    else (fired[0] if fired else ("truncated" if truncated else "unknown"))
                )
            )
        else:
            ended = (
                "object_dropped"
                if "object_dropped" in fired
                else ("time_out" if "time_out" in fired else (fired[0] if fired else "unknown"))
            )
        start = rows["apple_pos"][0]
        rec = {
            "episode": ep,
            "env": i,
            "mode": self.mode,
            "steps": int(len(succ)),
            "terms_fired": fired,
            "ended_by": ended,
            "arena_success": bool(succ.any()),
            "arena_success_step": first,
            "apple_start": start.tolist(),
            "apple_end": rows["apple_pos"][-1].tolist(),
            "plate_end": rows["plate_pos"][-1].tolist(),
            "apple_max_displacement_m": float(
                np.linalg.norm(rows["apple_pos"] - start, axis=1).max()
            ),
            "max_apple_lift_m": float((rows["apple_pos"][:, 2] - start[2]).max()),
            "end_xy_dist_to_plate_m": float(
                np.linalg.norm(rows["apple_pos"][-1, :2] - rows["plate_pos"][-1, :2])
            ),
            "wall_s": time.time() - self.ep_start[i],
        }
        if first is not None:
            rec["at_success"] = {
                "plate_force_n": float(rows["plate_force"][first]),
                "other_force_n": float(rows["other_force"][first]),
                "apple_speed_m_s": float(np.linalg.norm(rows["apple_vel"][first])),
                "hand_dist_m": float(rows["hand_dist"][first]),
                "xy_dist_to_plate_m": float(
                    np.linalg.norm(rows["apple_pos"][first, :2] - rows["plate_pos"][first, :2])
                ),
            }
        if self.mode == "settle":
            rec["strict"] = strict_verdict(rows, ended)
            rec["end_hand_dist_m"] = float(rows["hand_dist"][-1])
        self.written += 1
        self.ep_file.write(json.dumps(rec) + "\n")
        self.ep_file.flush()
        strict = rec.get("strict", {}).get("at_rest")
        print(
            f"[gr00t-baseline] episode {ep} env {i}: {ended}, arena_success={rec['arena_success']}"
            f" (step {first}), strict={strict}, {rec['steps']} steps, {rec['wall_s']:.1f} s",
            flush=True,
        )

    def close(self, metrics):
        self.ep_file.close()
        lat = np.asarray(self.latencies)
        summary = {
            "mode": self.mode,
            "episodes_recorded": self.written,
            "arena_metrics": metrics,
            "gr00t_get_action_calls": int(lat.size),
            "gr00t_latency_s": (
                {
                    "median": float(np.median(lat)),
                    "p95": float(np.percentile(lat, 95)),
                    "max": float(lat.max()),
                }
                if lat.size
                else None
            ),
            "wall_s": time.time() - self.t0,
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"[gr00t-baseline] summary {json.dumps(summary, default=str)}", flush=True)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    rargv = runner_argv(args)
    sys.path.insert(0, "/oej/src")
    from embodied_jepa.arena_transport import pin_pxr_work_thread_limit

    print(f"[arena] PXR_WORK_THREAD_LIMIT pinned to {pin_pxr_work_thread_limit()}", flush=True)
    import os

    os.chdir(ARENA_ROOT)
    sys.path.insert(0, ARENA_ROOT)
    sys.stdout.reconfigure(line_buffering=True)
    if os.environ.get("GROOT_DEPS_DIR"):
        # the runner's ensure_groot_deps_in_path() would re-exec the runner itself, bypassing
        # this guard; the local image does not set it (checked), so refuse rather than re-exec.
        sys.exit("GROOT_DEPS_DIR is set: refusing (a re-exec would bypass the client guard)")

    rec = _Recorder(args.output, args.mode, args.frame_episodes)
    from gr00t.policy import server_client

    guard_client(server_client.PolicyClient, rec.latencies)
    import gymnasium as gym
    import isaaclab_arena.evaluation.policy_runner as pr
    from isaaclab_arena.policy.policy_base import PolicyBase

    PolicyBase.shutdown_remote = lambda self, kill_server=False: None  # never kill

    class Recorded(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._obs = None
            self._wrap_success()

        def _wrap_success(self):
            tm = self.env.unwrapped.termination_manager
            cfg = tm.get_term_cfg("success")
            inner = cfg.func

            def success(env, **params):
                value = inner(env, **params)
                rec.capture(env, value, params)
                if args.mode == "settle":
                    import torch

                    return torch.zeros_like(value)  # observed, never terminating
                return value

            cfg.func = success

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            self._obs = obs
            return obs, info

        def step(self, action):
            prev = self._obs
            obs, rew, terminated, truncated, info = self.env.step(action)
            rec.after_step(self.env, obs, prev, terminated, truncated)
            self._obs = obs
            return obs, rew, terminated, truncated, info

    orig = pr.rollout_policy
    result = {}

    def rollout_policy(env, policy, num_steps, num_episodes, language_instruction=None):
        metrics = orig(Recorded(env), policy, num_steps, num_episodes, language_instruction)
        result["metrics"] = metrics
        return metrics

    pr.rollout_policy = rollout_policy
    sys.argv = [RUNNER, *rargv]
    print(f"[gr00t-baseline] runner argv: {rargv}", flush=True)
    (rec.out / "runner_argv.json").write_text(json.dumps(rargv))
    try:
        pr.main()
    finally:
        from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types

        m = result.get("metrics")
        rec.close(metrics_to_plain_python_types(m) if m is not None else None)
        print("[gr00t-baseline] policy_runner returned", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
