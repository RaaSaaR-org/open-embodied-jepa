"""D-RATE′'s two guards (R27.109): the run tree as working directory, and a model-load preflight
before the frozen runner (which claims the ledger) is called. CPU only; no run tree needed."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def drate():
    spec = importlib.util.spec_from_file_location(
        "dev_task092_drate", ROOT / "scripts" / "dev_task092_drate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cwd = os.getcwd()
    yield mod
    os.chdir(cwd)


def _tree(tmp_path, with_lewm=True):
    tree = tmp_path / "run-tree"
    (tree / "third_party").mkdir(parents=True)
    if with_lewm:
        (tree / "third_party" / "le-wm").mkdir()
    return tree


def test_enters_the_run_tree(drate, tmp_path):
    tree = _tree(tmp_path)
    os.chdir(tmp_path)
    assert drate.enter_run_tree(tree) == tree.resolve()
    assert Path.cwd().resolve() == tree.resolve()


def test_refuses_without_lewm_source_under_the_run_tree(drate, tmp_path):
    with pytest.raises(drate.Refused, match="G-cwd"):
        drate.enter_run_tree(_tree(tmp_path, with_lewm=False))
    with pytest.raises(drate.Refused, match="G-cwd"):
        drate.enter_run_tree(tmp_path / "missing")


def test_preflight_loads_both_models(drate):
    seen = []
    out = drate.preflight(lambda family, seed: seen.append((family, seed)) or {"ok": True})
    assert seen == [("G-req", 87100), ("G-v1", 87100)]
    assert set(out) == {"M-req-87100", "M-v1-87100"}


@pytest.mark.parametrize("bad", ["G-req", "G-v1"])
def test_preflight_refuses_if_either_load_fails(drate, bad):
    def load(family, seed):
        if family == bad:
            raise RuntimeError("LeWM source missing")
        return {}

    with pytest.raises(drate.Refused, match="G-preflight"):
        drate.preflight(load)


@pytest.mark.parametrize("command", [["goals", "--output", "x"], ["run", "--arm", "M-req"]])
def test_runner_is_not_called_when_a_guard_refuses(drate, tmp_path, command):
    calls = []

    def runner(argv):
        calls.append(argv)  # the frozen runner is what claims ledger keys
        return 0

    def bad_load(family, seed):
        raise RuntimeError("LeWM source missing")

    with pytest.raises(drate.Refused, match="G-preflight"):
        drate.main(command, run_tree=_tree(tmp_path), load=bad_load, runner=runner)
    with pytest.raises(drate.Refused, match="G-cwd"):
        drate.main(command, run_tree=_tree(tmp_path / "b", False), load=lambda f, s: {},
                   runner=runner)  # fmt: skip
    assert calls == []
    assert (
        drate.main(command, run_tree=_tree(tmp_path / "c"), load=lambda f, s: {}, runner=runner)
        == 0
    )
    assert calls and calls[0][-2:] == (["--workers", "8"] if command[0] == "goals" else
                                       ["--workers", "6"])  # fmt: skip
    assert "dev-drate2" in calls[0]


def test_fresh_resets_do_not_reuse_spent_ones(drate):
    assert len(drate.SEEDS) == 32
    assert not set(drate.SEEDS) & set(drate.SPENT)


@pytest.mark.parametrize(
    "command",
    [["goals", "--output", "x"], ["run", "--arm", "M-v1", "--goals", "g", "--output", "x"]],
)
def test_relative_paths_are_resolved_against_the_callers_cwd(drate, tmp_path, command):
    caller = tmp_path / "caller"
    caller.mkdir()
    os.chdir(caller)
    calls = []
    drate.main(command, run_tree=_tree(tmp_path), load=lambda f, s: {}, runner=calls.append)
    argv = calls[0]
    assert argv[argv.index("--output") + 1] == str(caller.resolve() / "x")
    if "--goals" in argv:
        assert argv[argv.index("--goals") + 1] == str(caller.resolve() / "g")
    assert Path.cwd().resolve() == (tmp_path / "run-tree").resolve()
