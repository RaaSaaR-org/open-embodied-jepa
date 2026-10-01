"""Shared fixtures. Only opt-in helpers live here; nothing is autouse."""

import os
from pathlib import Path

import pytest

import embodied_jepa
from embodied_jepa import devices

# Git worktrees share one .venv whose editable install points at a single checkout's src.
# pyproject.toml puts this checkout's src first on sys.path; refuse to run if the package
# still resolves elsewhere, because the tests would then exercise another revision's code.
_SRC = Path(__file__).resolve().parents[1] / "src"
_IMPORTED = Path(embodied_jepa.__file__).resolve()
if not _IMPORTED.is_relative_to(_SRC):
    raise pytest.UsageError(
        f"embodied_jepa was imported from {_IMPORTED}, not from this checkout's {_SRC}. "
        "Run pytest from the repository root so its pythonpath setting applies."
    )
# Subprocesses that tests start (python -m embodied_jepa..., scripts/*.py) inherit the
# environment, not sys.path: give them the same src first.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_SRC), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
)


@pytest.fixture
def restore_determinism():
    """Undo the process-wide flags configure_determinism sets, so other tests are unaffected.

    The cuBLAS variable is left as it is: importing embodied_jepa.devices sets it.
    """
    torch = pytest.importorskip("torch")
    before = devices.determinism_state()
    yield
    torch.use_deterministic_algorithms(
        before["deterministic_algorithms"],
        warn_only=before["deterministic_algorithms_warn_only"],
    )
    torch.backends.cudnn.deterministic = before["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = before["cuda_matmul_allow_tf32"]
    torch.set_float32_matmul_precision(before["float32_matmul_precision"])
