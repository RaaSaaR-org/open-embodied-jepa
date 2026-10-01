"""New runner scripts must not load other scripts (the 2026-10-02 audit, F16).

The pinned runners chain into each other (``run_obs_ceiling_v2.py`` loads
``run_lewm_planner_v2.py``, which loads ``run_wm_critic_v2.py``, which loads
``run_first_policy_v2_m2.py`` ...), and MemoryWatch and the PSS code are copied between them.
They are pinned by sha256, so they stay as they are and are listed below with exactly the scripts
they load. Any other ``scripts/**/run_*.py`` must take shared code from ``src/embodied_jepa``
(``run_guards``, ``run_tools``) and may not load a script file, by path or by module name."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# runner -> the scripts it loads, as of c16fb04. Never add to this list; new runners use
# embodied_jepa.run_tools instead.
PINNED_EDGES = {
    "scripts/run_first_policy.py": {"evaluate_apple.py", "train_apple_latent_dynamics.py"},
    "scripts/run_first_policy_v2.py": {
        "collect_apple_wide.py",
        "evaluate_apple.py",
        "train_apple_latent_dynamics.py",
    },
    "scripts/run_first_policy_v2_linux.py": {
        "collect_apple_wide.py",
        "evaluate_apple.py",
        "train_apple_latent_dynamics.py",
    },
    "scripts/run_first_policy_v2_m2.py": {"run_first_policy_v2_linux.py"},
    "scripts/run_lewm_planner_v2.py": {"run_wm_critic_v2.py"},
    "scripts/run_obs_ceiling_v2.py": {"run_lewm_planner_v2.py"},
    "scripts/run_v2_expert_gate.py": {"evaluate_apple.py"},
    "scripts/run_wm_critic_v2.py": {"run_first_policy_v2_m2.py"},
}
LOADERS = {"spec_from_file_location", "SourceFileLoader", "run_path", "run_module"}


def _script_names() -> set[str]:
    return {p.name for p in SCRIPTS.rglob("*.py")}


def script_loads(path: Path) -> tuple[set[str], list[str]]:
    """The script files ``path`` names (string literals ending in a script's file name, and
    imports of a script's module name), and any loader calls it makes."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = _script_names() - {path.name}
    stems = {n[:-3] for n in names}
    loaded: set[str] = set()
    loaders: list[str] = []
    docstrings = {  # prose, not a load: bare string statements
        id(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            tail = node.value.replace("\\", "/").rsplit("/", 1)[-1]
            if tail in names:
                loaded.add(tail)
        elif isinstance(node, ast.Import):
            loaded |= {f"{a.name.split('.')[0]}.py" for a in node.names if a.name in stems}
        elif isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")[0]
            if head in stems or head == "scripts":
                loaded.add(f"{node.module.split('.')[-1]}.py")
        elif isinstance(node, ast.Attribute) and node.attr in LOADERS:
            loaders.append(node.attr)
        elif isinstance(node, ast.Name) and node.id in LOADERS:
            loaders.append(node.id)
    return loaded, loaders


def _runners() -> list[Path]:
    return sorted(SCRIPTS.rglob("run_*.py"))


def test_no_new_runner_loads_another_script():
    offenders = {}
    for path in _runners():
        rel = path.relative_to(ROOT).as_posix()
        loaded, loaders = script_loads(path)
        if rel in PINNED_EDGES:
            extra = loaded - PINNED_EDGES[rel]
            if extra:
                offenders[rel] = f"loads more than its pinned edges: {sorted(extra)}"
        elif loaded or loaders:
            offenders[rel] = f"loads {sorted(loaded)} via {sorted(set(loaders))}"
    assert not offenders, (
        "runner scripts must not load other scripts; move shared code to src/embodied_jepa "
        f"(run_tools): {offenders}"
    )


def test_the_pinned_edges_are_still_accurate():
    """The allowlist names what the pinned runners really load (so it cannot silently cover a
    deleted or renamed runner)."""
    for rel, edges in PINNED_EDGES.items():
        loaded, loaders = script_loads(ROOT / rel)
        assert loaded == edges, rel
        assert loaders, rel


def test_the_detector_catches_each_form(tmp_path, monkeypatch):
    fake = tmp_path / "scripts"
    fake.mkdir()
    (fake / "run_old.py").write_text("")
    (fake / "helper.py").write_text("")
    monkeypatch.setattr(sys.modules[__name__], "SCRIPTS", fake)
    cases = {
        'spec = importlib.util.spec_from_file_location("x", "run_old.py")\n': (
            {"run_old.py"},
            True,
        ),
        "import run_old\n": ({"run_old.py"}, False),
        "from helper import thing\n": ({"helper.py"}, False),
        "import runpy\nrunpy.run_path(p)\n": (set(), True),
        "from embodied_jepa import run_tools\n": (set(), False),
        '"""Unlike run_old.py, this runner loads nothing."""\n': (set(), False),
    }
    for source, (want, has_loader) in cases.items():
        new = fake / "run_new.py"
        new.write_text(source)
        loaded, loaders = script_loads(new)
        assert loaded == want, source
        assert bool(loaders) == has_loader, source
