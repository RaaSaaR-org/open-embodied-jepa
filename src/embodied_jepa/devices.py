"""One place for torch device names, availability, CUDA determinism, sync and memory.

The supported devices are ``cpu``, ``mps`` (Apple silicon) and ``cuda`` (one NVIDIA GPU, the
current device). ``auto`` resolves to the first available of cuda, mps, cpu.

Selecting ``cuda`` through :func:`require` (or :func:`configure_determinism` directly) applies
the project's single deterministic CUDA setup, process-wide:

- ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (set if absent; refused if set to anything else);
- ``torch.use_deterministic_algorithms(True, warn_only=not strict)``;
- ``torch.backends.cudnn.deterministic = True`` and ``benchmark = False``;
- TF32 off: ``torch.set_float32_matmul_precision("highest")`` and both ``allow_tf32`` flags.

That selects the deterministic kernel of every operation that has one. It does not make CUDA
results bit-identical to CPU or MPS results, or across machines, drivers or library versions.

``strict`` decides what happens at an operation PyTorch has no deterministic CUDA kernel for. The
default (``strict=False``) warns and runs it: ``native_jepa``'s ``AdaptiveAvgPool2d`` backward is
such an operation (``adaptive_avg_pool2d_backward_cuda``), so strict mode would make that backend
untrainable on CUDA. In the same mode PyTorch also warns that LeWM's causal attention backward
(the memory-efficient scaled-dot-product kernel) "defaults to a non-deterministic algorithm";
strict mode would select its deterministic variant. The attention backends are left as PyTorch
chooses them: forcing the math backend fails at sequence length 1 on torch 2.14 without a C
compiler (it JIT-builds a Triton helper), and the Linux PC has none. Same-seed reproducibility is
therefore an empirical property here, checked by ``scripts/cuda_smoke.py`` and by any protocol
that relies on it. A protocol whose operations all have deterministic kernels can pass
``strict=True`` so that a nondeterministic one raises.
``torch`` is imported lazily, so importing this module does not import torch.
"""

from __future__ import annotations

import os

SUPPORTED_DEVICES = ("cpu", "mps", "cuda")
AUTO_ORDER = ("cuda", "mps", "cpu")
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class DeviceError(ValueError):
    """A device name is unknown, or the requested device is unavailable."""


def available(device: str) -> bool:
    """Whether ``device`` can be used in this process (unknown names are unavailable)."""
    if device == "cpu":
        return True
    if device not in SUPPORTED_DEVICES:
        return False
    import torch

    if device == "mps":
        return bool(torch.backends.mps.is_available())
    return bool(torch.cuda.is_available())


def resolve(device: str, *, allow_auto: bool = True) -> str:
    """Validate ``device`` (and resolve ``auto``); raise :class:`DeviceError` if unusable."""
    if device == "auto" and allow_auto:
        return next(name for name in AUTO_ORDER if available(name))
    if device not in SUPPORTED_DEVICES:
        choices = SUPPORTED_DEVICES + (("auto",) if allow_auto else ())
        raise DeviceError(f"device must be one of {', '.join(choices)}; got {device!r}")
    if not available(device):
        raise DeviceError(f"requested device {device!r} is unavailable")
    return device


def require(
    device: str,
    *,
    allow_auto: bool = False,
    error: type[Exception] = DeviceError,
    strict: bool = False,
) -> str:
    """:func:`resolve`, then apply the deterministic setup when the result is ``cuda``.

    ``error`` lets a caller keep its own exception type (for example ``ContractError``).
    """
    try:
        name = resolve(device, allow_auto=allow_auto)
    except DeviceError as exc:
        if error is DeviceError:
            raise
        raise error(str(exc)) from exc
    if name == "cuda":
        configure_determinism(name, strict=strict)
    return name


def configure_determinism(device: str, *, strict: bool = False) -> dict:
    """Apply the deterministic CUDA setup (no-op for cpu/mps); return :func:`determinism_state`."""
    if device != "cuda":
        return determinism_state()
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is None:
        import torch

        if torch.cuda.is_initialized():
            # PyTorch reads the variable once, when it first sizes the cuBLAS workspace.
            raise DeviceError(
                "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was set; export "
                f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG} before starting the process"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    elif current != CUBLAS_WORKSPACE_CONFIG:
        raise DeviceError(
            f"CUBLAS_WORKSPACE_CONFIG is {current!r}; the deterministic setup requires "
            f"{CUBLAS_WORKSPACE_CONFIG!r}"
        )
    import torch

    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return determinism_state()


def determinism_state() -> dict:
    """The process-wide settings :func:`configure_determinism` controls, for run reports."""
    import torch

    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdp_backends": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
        },
    }


def synchronize(device: str) -> None:
    """Wait for queued kernels on an accelerator, so timings include them (no-op on cpu)."""
    if device == "cpu":
        return
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory_report(device: str) -> dict:
    """Accelerator memory counters, keyed by device so they are never mistaken for host RSS."""
    import torch

    if device == "mps":
        return {
            "final_mps_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "final_mps_driver_bytes": int(torch.mps.driver_allocated_memory()),
        }
    if device == "cuda":
        return {
            "final_cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "final_cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    return {}


def accelerator_info(device: str) -> dict:
    """Identifying details of the accelerator behind ``device``, for run reports."""
    import torch

    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return {
            "cuda_device_name": properties.name,
            "cuda_capability": list(torch.cuda.get_device_capability(index)),
            "cuda_total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    if device == "mps":
        return {"mps_available": bool(torch.backends.mps.is_available())}
    return {}


def get_rng_state(device: str):
    """The accelerator generator state for ``device`` (``None`` on cpu)."""
    import torch

    if device == "mps":
        return torch.mps.get_rng_state()
    if device == "cuda":
        return torch.cuda.get_rng_state()
    return None


def set_rng_state(device: str, state) -> None:
    import torch

    if device == "mps":
        torch.mps.set_rng_state(state)
    elif device == "cuda":
        torch.cuda.set_rng_state(state)


def manual_seed(device: str, seed: int) -> None:
    """Seed only the accelerator generator of ``device`` (the CPU generator is untouched)."""
    import torch

    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed(seed)
