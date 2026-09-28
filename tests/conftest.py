"""Shared fixtures. Only opt-in helpers live here; nothing is autouse."""

import pytest

from embodied_jepa import devices


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
