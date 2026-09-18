"""Capture the hardware/software context of a run. A latency number without
its machine, OS and engine version is not a result."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _run(cmd: list[str]) -> str | None:
    """stdout of a command, or None if it is missing or fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return (r.stdout or r.stderr).strip() or None


def collect() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if info["system"] == "Darwin":
        info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem and mem.isdigit() else None
        info["macos"] = _run(["sw_vers", "-productVersion"])
        pmset = _run(["pmset", "-g"]) or ""
        # Low Power Mode throttles the SoC and silently ruins a benchmark.
        info["low_power_mode"] = any(
            line.split()[:1] == ["lowpowermode"] and line.split()[-1] == "1"
            for line in pmset.splitlines()
        )
        info["on_ac_power"] = "AC Power" in (_run(["pmset", "-g", "batt"]) or "")
    elif info["system"] == "Linux":
        try:
            with open("/proc/meminfo") as fh:
                info["ram_gb"] = round(int(fh.readline().split()[1]) / 2**20, 1)
        except (OSError, ValueError, IndexError):
            info["ram_gb"] = None
        try:
            with open("/proc/cpuinfo") as fh:
                names = [
                    line.split(":", 1)[1].strip()
                    for line in fh
                    if line.lower().startswith(("model name", "hardware"))
                ]
            info["chip"] = names[0] if names else None
        except OSError:
            info["chip"] = None
    if shutil.which("nvidia-smi"):
        info["gpu"] = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        )
    for tool in ("llama-server", "ollama", "macmon"):
        exe = shutil.which(tool)
        if exe:
            version = _run([exe, "--version"])
            info[f"tool:{tool}"] = version.splitlines()[-1] if version else "present"
        else:
            info[f"tool:{tool}"] = None
    info["git_commit"] = _run(["git", "rev-parse", "--short", "HEAD"]) if shutil.which("git") else None
    return info
