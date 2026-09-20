"""Bounded CPU/MPS readiness probe; no model weights or datasets downloaded."""

import json
import platform
import resource
import shutil
import subprocess
from pathlib import Path

import torch


def probe(device):
    torch.manual_seed(0)
    x = torch.randn(32, 16, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    return {
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_gradient": bool(torch.isfinite(x.grad).all().item()),
    }


if __name__ == "__main__":
    torch.set_num_threads(4)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "disk_free_bytes": shutil.disk_usage(".").free,
        "torch": torch.__version__,
        "cpu": probe("cpu"),
        "mps_available": torch.backends.mps.is_available(),
        "mps": probe("mps") if torch.backends.mps.is_available() else None,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    dest = Path("outputs/feasibility/resources.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
