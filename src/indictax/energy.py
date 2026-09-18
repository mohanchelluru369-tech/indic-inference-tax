"""Power sampling and energy integration.

A sampler runs a platform tool in the background, timestamps every reading
with time.monotonic() (the same clock the client uses), and integrates power
over a request's [t_start, t_end] window to get joules.

    macmon        Apple Silicon, no sudo      brew install macmon     (preferred on the Mac)
    powermetrics  Apple Silicon, needs sudo   run `sudo -v` first
    nvidia-smi    NVIDIA GPUs (board power only, excludes CPU/host)
    none          timing only

Absolute joules are NOT comparable across platforms: the Mac channels cover
the SoC, nvidia-smi covers only the GPU board. The claims this project makes
are within-platform ratios between languages. See docs/03_methodology.md.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    t: float
    channels: dict[str, float]


class PowerSampler:
    name = "none"
    primary = "w"

    def __init__(self, interval_ms: int = 250):
        self.interval_ms = interval_ms
        self.samples: list[Sample] = []
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def max_gap_s(self) -> float:
        """Longest silence between readings that is still treated as coverage."""
        return max(2.0, 8 * self.interval_ms / 1000.0)

    # --- subclass hooks -------------------------------------------------
    def command(self) -> list[str] | None:
        return None

    def parse(self, line: str) -> dict[str, float] | None:
        return None

    # --- lifecycle ------------------------------------------------------
    def start(self) -> None:
        cmd = self.command()
        if not cmd:
            return
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
            )
        except OSError as e:
            self.error = f"cannot start {cmd[0]}: {e}"
            return
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            now = time.monotonic()
            try:
                channels = self.parse(line)
            except Exception as e:  # noqa: BLE001 - never let a parse error kill the run
                self.error = f"parse error: {e}"
                continue
            if channels:
                with self._lock:
                    self.samples.append(Sample(now, channels))
        # stdout closed: the tool exited. If it never produced a reading, keep its
        # own error message so `indictax doctor` can show what went wrong.
        try:
            code = self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return
        if code != 0 and not self.samples:
            err = (self._proc.stderr.read() if self._proc.stderr else "") or ""
            last = err.strip().splitlines()[-1] if err.strip() else ""
            self.error = f"{self.name} exited with code {code}: {last[:200]}"

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)

    # --- queries --------------------------------------------------------
    def _series(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            pts = [(s.t, s.channels[channel]) for s in self.samples if channel in s.channels]
        if not pts:
            return np.array([]), np.array([])
        t, w = zip(*pts)
        return np.array(t), np.array(w)

    def energy_j(self, t0: float, t1: float, channel: str | None = None) -> float | None:
        """Trapezoidal integral of power over [t0, t1], interpolating at the edges."""
        t, w = self._series(channel or self.primary)
        return integrate(t, w, t0, t1, self.max_gap_s)

    def mean(self, t0: float, t1: float, channel: str) -> float | None:
        t, w = self._series(channel)
        if t.size == 0 or t1 <= t0:
            return None
        e = integrate(t, w, t0, t1, self.max_gap_s)
        return None if e is None else e / (t1 - t0)

    def idle_baseline_w(self, seconds: float, channel: str | None = None) -> float | None:
        """Block for `seconds` with nothing running and return mean power."""
        t0 = time.monotonic()
        time.sleep(seconds)
        return self.mean(t0, time.monotonic(), channel or self.primary)


def integrate(t: np.ndarray, w: np.ndarray, t0: float, t1: float, max_gap: float = 2.0) -> float | None:
    """Trapezoidal energy over [t0, t1], or None unless the window is actually covered.

    np.interp clamps outside the sampled range and draws a straight line across
    holes, so without these checks a sampler that died mid-run would keep
    "producing" joules. Required: a reading within max_gap of each edge, and no
    gap longer than max_gap between consecutive readings in the window."""
    if t.size == 0 or t1 <= t0:
        return None
    near = t[(t >= t0 - max_gap) & (t <= t1 + max_gap)]
    if near.size == 0:
        return None
    if near[0] > t0 + max_gap or near[-1] < t1 - max_gap:
        return None
    if near.size > 1 and float(np.max(np.diff(near))) > max_gap:
        return None
    inside = (t > t0) & (t < t1)
    grid = np.concatenate(([t0], t[inside], [t1]))
    power = np.interp(grid, t, w)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(power, grid))


class MacmonSampler(PowerSampler):
    """https://github.com/vladkens/macmon - JSON lines, watts, no sudo."""

    name = "macmon"
    primary = "soc_w"

    def command(self) -> list[str] | None:
        exe = shutil.which("macmon")
        if not exe:
            self.error = "macmon not found (brew install macmon)"
            return None
        return [exe, "pipe", "-s", "0", "-i", str(self.interval_ms)]

    def parse(self, line: str) -> dict[str, float] | None:
        line = line.strip()
        if not line.startswith("{"):
            return None
        d = json.loads(line)
        out: dict[str, float] = {}
        if "all_power" in d:
            out["soc_w"] = float(d["all_power"]) + float(d.get("ram_power", 0.0))
        if "sys_power" in d:
            out["sys_w"] = float(d["sys_power"])
        for k in ("cpu_power", "gpu_power", "ane_power", "ram_power"):
            if k in d:
                out[k.replace("_power", "_w")] = float(d[k])
        temp = d.get("temp") or {}
        if "gpu_temp_avg" in temp:
            out["gpu_temp_c"] = float(temp["gpu_temp_avg"])
        if "cpu_temp_avg" in temp:
            out["cpu_temp_c"] = float(temp["cpu_temp_avg"])
        return out or None


class PowermetricsSampler(PowerSampler):
    name = "powermetrics"
    primary = "soc_w"
    _combined = re.compile(r"Combined Power.*?:\s*([\d.]+)\s*mW")
    _part = re.compile(r"^(CPU|GPU|ANE) Power:\s*([\d.]+)\s*mW")

    def command(self) -> list[str] | None:
        if not shutil.which("powermetrics"):
            self.error = "powermetrics not found (macOS only)"
            return None
        # -n: never prompt. Run `sudo -v` in the same terminal before the benchmark.
        return ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
                "-i", str(self.interval_ms)]

    def parse(self, line: str) -> dict[str, float] | None:
        m = self._combined.search(line)
        if m:
            return {"soc_w": float(m.group(1)) / 1000.0}
        m = self._part.match(line.strip())
        if m:
            return {f"{m.group(1).lower()}_w": float(m.group(2)) / 1000.0}
        return None


class NvidiaSmiSampler(PowerSampler):
    name = "nvidia-smi"
    primary = "gpu_w"

    def command(self) -> list[str] | None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            self.error = "nvidia-smi not found"
            return None
        return [exe, "-i", "0", "--query-gpu=power.draw,temperature.gpu",
                "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)]

    def parse(self, line: str) -> dict[str, float] | None:
        parts = [p.strip() for p in line.split(",")]
        try:
            out = {"gpu_w": float(parts[0])}
        except (ValueError, IndexError):
            return None
        if len(parts) > 1:
            try:
                out["gpu_temp_c"] = float(parts[1])
            except ValueError:
                pass
        return out


SAMPLERS = {
    "none": PowerSampler,
    "macmon": MacmonSampler,
    "powermetrics": PowermetricsSampler,
    "nvidia-smi": NvidiaSmiSampler,
}


def make_sampler(kind: str = "auto", interval_ms: int = 250) -> PowerSampler:
    if kind == "auto":
        if shutil.which("macmon"):
            kind = "macmon"
        elif shutil.which("nvidia-smi"):
            kind = "nvidia-smi"
        else:
            kind = "none"
    if kind not in SAMPLERS:
        raise ValueError(f"unknown power sampler {kind!r}; choose from {sorted(SAMPLERS)} or 'auto'")
    return SAMPLERS[kind](interval_ms=interval_ms)


@dataclass
class SamplerCheck:
    name: str
    ok: bool
    detail: str
    readings: int = 0
    example: dict = field(default_factory=dict)


def self_check(kind: str = "auto", seconds: float = 2.0) -> SamplerCheck:
    """Start the sampler briefly and report whether real readings arrive."""
    s = make_sampler(kind)
    if s.name == "none":
        return SamplerCheck("none", False, "no power tool found; runs will record timing only")
    s.start()
    time.sleep(seconds)
    s.stop()
    if s.samples:
        return SamplerCheck(s.name, True, f"{len(s.samples)} readings in {seconds:.0f}s",
                            len(s.samples), s.samples[-1].channels)
    return SamplerCheck(s.name, False, s.error or "started but produced no readings")
