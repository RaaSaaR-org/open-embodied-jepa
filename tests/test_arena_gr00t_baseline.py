"""Host tests for scripts/isaac/arena_gr00t_baseline.py (no Isaac, no GR00T server)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/isaac/arena_gr00t_baseline.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("arena_gr00t_baseline", PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_runner_argv_is_the_tutorial_command(mod):
    args = mod.parse_args(["--output", "/o", "--num_steps", "600"])
    assert mod.runner_argv(args) == [
        "--headless",
        "--policy_type",
        "isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy",
        "--policy_config_yaml_path",
        "isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml",
        "--remote_host",
        "localhost",
        "--remote_port",
        "5555",
        "--num_steps",
        "600",
        "--enable_cameras",
        "galileo_g1_static_pick_and_place",
        "--object",
        "apple_01_objaverse_robolab",
        "--destination",
        "clay_plates_hot3d_robolab",
        "--embodiment",
        "g1_wbc_agile_joint",
    ]
    args = mod.parse_args(["--output", "/o", "--num_episodes", "30", "--num_envs", "2"])
    argv = mod.runner_argv(args)
    assert argv[argv.index("--num_episodes") + 1] == "30"
    assert argv[argv.index("--num_envs") + 1] == "2"
    assert not any("kill" in a or "exit" in a for a in argv)


@pytest.mark.parametrize(
    "bad",
    [
        ["--remote_kill_on_exit"],
        ["--kill"],
        ["--exit"],
        ["--remote_host", "10.0.0.2"],
        ["--remote_port", "5556"],
        ["--host", "x"],
        ["--port", "5556"],
        ["--policy_type", "zero_action"],
        ["--policy_config_yaml_path", "x.yaml"],
        ["--shutdown"],
        ["--unknown_flag"],
        ["--num_steps", "0"],
    ],
)
def test_refuses_kill_exit_host_port_policy_and_unknown_flags(mod, bad):
    base = ["--output", "/o", "--num_episodes", "3"]
    if bad[0] == "--num_steps":
        base = ["--output", "/o"]
    with pytest.raises(SystemExit):
        mod.parse_args(base + bad)


def test_needs_exactly_one_length(mod):
    with pytest.raises(SystemExit):
        mod.parse_args(["--output", "/o"])
    with pytest.raises(SystemExit):
        mod.parse_args(["--output", "/o", "--num_steps", "6", "--num_episodes", "2"])


def test_check_runner_argv_refuses_tampering(mod):
    good = mod.runner_argv(mod.parse_args(["--output", "/o", "--num_steps", "600"]))
    mod.check_runner_argv(good)
    for bad in (
        good + ["--remote_kill_on_exit"],
        [("127.0.0.2" if a == "localhost" else a) for a in good],
        [("5556" if a == "5555" else a) for a in good],
        [("zero_action" if a == mod.POLICY_TYPE else a) for a in good],
        good + ["--remote_port", "5555"],
    ):
        with pytest.raises(SystemExit):
            mod.check_runner_argv(bad)


class FakeClient:
    def __init__(self):
        self.calls = []
        self._init_socket()

    def _init_socket(self):
        self.socket = None

    def call_endpoint(self, endpoint, data=None, requires_input=True):
        self.calls.append(endpoint)
        return {"ok": endpoint}

    def kill_server(self):
        self.call_endpoint("kill", requires_input=False)


def test_client_guard_blocks_kill_and_unknown_endpoints(mod):
    lat = []
    cls = mod.guard_client(type("C", (FakeClient,), {}), lat, timeout_ms=10)
    cls._init_socket = FakeClient._init_socket  # no zmq socket in the fake
    c = cls()
    for ok in ("ping", "get_action", "reset", "get_modality_config"):
        assert c.call_endpoint(ok) == {"ok": ok}
    assert len(lat) == 1  # only get_action is timed
    with pytest.raises(PermissionError):
        c.kill_server()
    with pytest.raises(PermissionError):
        c.call_endpoint("kill", requires_input=False)
    with pytest.raises(PermissionError):
        c.call_endpoint("anything_else")
    assert "kill" not in c.calls
    assert mod.guard_client(cls) is cls  # idempotent


def _rows(t, *, apple_xy=(0.0, 0.0), speed=0.0, plate_force=1.0, other=0.0):
    apple = np.zeros((t, 3))
    apple[:, :2] = apple_xy
    vel = np.zeros((t, 3))
    vel[:, 0] = speed
    return {
        "apple_pos": apple,
        "plate_pos": np.zeros((t, 3)),
        "apple_vel": vel,
        "plate_force": np.full(t, plate_force),
        "other_force": np.full(t, other),
    }


def test_strict_verdict(mod):
    assert mod.WINDOW_STEPS == 50
    assert mod.strict_verdict(_rows(300), "time_out")["at_rest"]
    assert not mod.strict_verdict(_rows(300), "object_dropped")["at_rest"]
    assert not mod.strict_verdict(_rows(30), "time_out")["at_rest"]
    v = mod.strict_verdict(_rows(300, apple_xy=(0.041, 0.0)), "time_out")
    assert not v["at_rest"] and not v["inside_all"]
    v = mod.strict_verdict(_rows(300, speed=0.002), "time_out")
    assert not v["at_rest"] and not v["still_all"]
    v = mod.strict_verdict(_rows(300, plate_force=0.4), "time_out")
    assert not v["at_rest"] and not v["supported_all"]
    v = mod.strict_verdict(_rows(300, other=0.5), "time_out")
    assert not v["at_rest"] and not v["released_all"]
    # only the final 50 steps count
    r = _rows(300)
    r["apple_vel"][:250, 0] = 1.0
    assert mod.strict_verdict(r, "time_out")["at_rest"]
    r["apple_vel"][250, 0] = 1.0
    assert not mod.strict_verdict(r, "time_out")["at_rest"]


def test_policy_runner_wrapper_still_refuses_gr00t():
    spec = importlib.util.spec_from_file_location(
        "arena_policy_runner", PATH.parent / "arena_policy_runner.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with pytest.raises(SystemExit):
        m.check_args(["--policy_type", "isaaclab_arena_gr00t.policy.X", "--remote_port", "5555"])
