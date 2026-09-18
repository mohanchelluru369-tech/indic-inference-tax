import numpy as np
import pytest

from indictax.energy import (MacmonSampler, NvidiaSmiSampler, PowermetricsSampler, PowerSampler,
                             Sample, integrate, make_sampler)


def test_constant_power_integrates_to_watts_times_seconds():
    t = np.arange(0.0, 10.0, 0.25)
    w = np.full_like(t, 8.0)
    assert integrate(t, w, 2.0, 5.0) == pytest.approx(24.0)


def test_window_edges_are_interpolated_not_snapped():
    # Power ramps 0 -> 10 W over 10 s; energy over [2.1, 4.3] is the exact trapezoid area.
    t = np.arange(0.0, 10.5, 0.5)
    w = t.copy()
    assert integrate(t, w, 2.1, 4.3) == pytest.approx((4.3**2 - 2.1**2) / 2, rel=1e-6)


def test_no_readings_near_window_returns_none_rather_than_a_guess():
    t = np.array([0.0, 0.25, 0.5])
    w = np.array([5.0, 5.0, 5.0])
    assert integrate(t, w, 100.0, 101.0) is None
    assert integrate(np.array([]), np.array([]), 0.0, 1.0) is None
    assert integrate(t, w, 1.0, 1.0) is None


def test_sampler_queries_use_the_primary_channel():
    s = PowerSampler()
    s.primary = "soc_w"
    s.samples = [Sample(float(i), {"soc_w": 4.0, "gpu_temp_c": 50.0 + i}) for i in range(11)]
    assert s.energy_j(0.0, 10.0) == pytest.approx(40.0)
    assert s.mean(0.0, 10.0, "gpu_temp_c") == pytest.approx(55.0)
    assert s.energy_j(0.0, 10.0, channel="missing") is None


def test_macmon_line():
    line = ('{"timestamp":"2026-09-17T10:00:00Z","temp":{"cpu_temp_avg":43.7,"gpu_temp_avg":36.9},'
            '"cpu_power":0.20,"gpu_power":6.5,"ane_power":0.0,"all_power":6.7,"sys_power":14.2,"ram_power":0.3}')
    ch = MacmonSampler().parse(line)
    assert ch["soc_w"] == pytest.approx(7.0)  # all_power + ram_power
    assert ch["sys_w"] == 14.2 and ch["gpu_w"] == 6.5 and ch["gpu_temp_c"] == 36.9
    assert MacmonSampler().parse("not json") is None


def test_powermetrics_lines():
    p = PowermetricsSampler()
    assert p.parse("Combined Power (CPU + GPU + ANE): 8123 mW") == {"soc_w": pytest.approx(8.123)}
    assert p.parse("GPU Power: 6000 mW") == {"gpu_w": 6.0}
    assert p.parse("*** Sampled system activity ***") is None


def test_nvidia_smi_lines():
    n = NvidiaSmiSampler()
    assert n.parse("45.32, 61") == {"gpu_w": 45.32, "gpu_temp_c": 61.0}
    assert n.parse("[N/A], 61") is None


def test_make_sampler():
    assert make_sampler("none").name == "none"
    with pytest.raises(ValueError):
        make_sampler("wattmeter-9000")


def test_live_subprocess_pump_with_a_fake_macmon(tmp_path, monkeypatch):
    """Runs the real start/pump/stop path against a stand-in `macmon` executable."""
    import os
    import stat
    import time

    from indictax.energy import self_check

    fake = tmp_path / "macmon"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        "assert sys.argv[1:] == ['pipe', '-s', '0', '-i', '50'], sys.argv\n"
        "while True:\n"
        "    print(json.dumps({'all_power': 5.0, 'ram_power': 1.0, 'sys_power': 12.0,\n"
        "                      'temp': {'gpu_temp_avg': 40.0}}), flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    s = make_sampler("auto", interval_ms=50)
    assert s.name == "macmon"
    s.start()
    # Wait for the first reading before opening the window: spawning a Python
    # subprocess takes ~0.5 s on macOS, and the window must start inside coverage.
    deadline = time.monotonic() + 5.0
    while not s.samples and time.monotonic() < deadline:
        time.sleep(0.02)
    assert s.samples, s.error
    t0 = time.monotonic()
    time.sleep(0.6)
    t1 = time.monotonic()
    s.stop()
    assert len(s.samples) >= 5 and s.error is None
    assert s.energy_j(t0, t1) == pytest.approx(6.0 * (t1 - t0), rel=0.02)  # 6 W constant

    # self_check uses the default 250 ms interval, which the stand-in rejects and
    # exits on: the failure must surface with the tool's own message, not silently.
    # The budget is generous because a loaded CI runner can take a second just to
    # spawn Python; self_check returns as soon as the failure is visible, so the
    # generous budget costs nothing when things work.
    t0 = time.monotonic()
    chk = self_check("macmon", seconds=20.0)
    assert time.monotonic() - t0 < 15.0, "self_check waited out the budget instead of returning early"
    assert chk.ok is False
    assert "exited with code 1" in chk.detail and "AssertionError" in chk.detail


def test_a_dead_or_stalled_sampler_yields_none_not_invented_joules():
    t = np.arange(0.0, 10.25, 0.25)          # sampler dies at t = 10 s
    w = np.full_like(t, 20.0)
    assert integrate(t, w, 2.0, 8.0) == pytest.approx(120.0)     # covered
    assert integrate(t, w, 9.0, 69.0) is None                    # died 1 s into a 60 s request
    assert integrate(t, w, 11.9, 71.9) is None                   # request entirely after death
    late = t + 100.0
    assert integrate(late, w, 38.0, 98.1) is None                # request before the sampler started
    holed = np.concatenate([t[t <= 5], t[t >= 35] ])             # 30 s stall in the middle
    assert integrate(holed, np.full_like(holed, 20.0), 4.0, 36.0) is None
    # a request shorter than the sampling interval is still fine if readings surround it
    assert integrate(t, w, 3.30, 3.40) == pytest.approx(2.0)


def test_self_check_returns_as_soon_as_readings_arrive(monkeypatch):
    """A healthy sampler must not cost `doctor` the whole budget."""
    import time

    from indictax.energy import self_check

    class Fast(PowerSampler):
        name = "fake"
        primary = "soc_w"

        def start(self):
            self.samples = [Sample(time.monotonic(), {"soc_w": 3.0}) for _ in range(4)]

        def stop(self):
            pass

    monkeypatch.setitem(__import__("indictax.energy", fromlist=["SAMPLERS"]).SAMPLERS, "fake", Fast)
    t0 = time.monotonic()
    chk = self_check("fake", seconds=30.0)
    assert time.monotonic() - t0 < 1.0
    assert chk.ok and chk.readings == 4 and chk.example == {"soc_w": 3.0}
