"""Bounded CPU/MPS/CUDA readiness probe; no model weights or datasets downloaded."""

import json
import platform
import resource
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from embodied_jepa import devices


def probe(device):
    torch.manual_seed(0)
    x = torch.randn(32, 16, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    devices.synchronize(device)
    return {
        "finite_loss": bool(torch.isfinite(loss).item()),
        "finite_gradient": bool(torch.isfinite(x.grad).all().item()),
    }


def host():
    """CPU model name and physical memory in bytes (macOS sysctl, Linux /proc)."""
    if sys.platform == "darwin":
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        return chip.strip(), memory
    chip, memory = platform.processor(), None
    cpuinfo, meminfo = Path("/proc/cpuinfo"), Path("/proc/meminfo")
    if cpuinfo.exists():
        names = [line for line in cpuinfo.read_text().splitlines() if line.startswith("model name")]
        if names:
            chip = names[0].split(":", 1)[1].strip()
    if meminfo.exists():
        totals = [line for line in meminfo.read_text().splitlines() if line.startswith("MemTotal")]
        if totals:
            memory = int(totals[0].split()[1]) * 1024
    return chip, memory


if __name__ == "__main__":
    torch.set_num_threads(4)
    chip, memory = host()
    mps, cuda = devices.available("mps"), devices.available("cuda")
    if cuda:
        devices.configure_determinism("cuda")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chip": chip,
        "memory_bytes": memory,
        "disk_free_bytes": shutil.disk_usage(".").free,
        "torch": torch.__version__,
        "cpu": probe("cpu"),
        "mps_available": mps,
        "mps": probe("mps") if mps else None,
        "cuda_available": cuda,
        "cuda": probe("cuda") | devices.accelerator_info("cuda") if cuda else None,
        # ru_maxrss is in bytes on macOS and in KiB on Linux.
        "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
    }
    dest = Path("outputs/feasibility/resources.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
