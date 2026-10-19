"""Log GPU clocks / temperature / power / P-state / throttle reasons via nvidia-smi.

Snapshot:   python scripts/gpu_monitor.py --once results/raw/gpu_state_idle.csv
Background: python scripts/gpu_monitor.py results/raw/gpu_log_sweep.csv --interval-ms 500
            (runs until killed; run_all starts/stops it around the sweep)
"""
import argparse, subprocess, sys
from pathlib import Path

FIELDS = ("timestamp,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.max.sm,"
          "utilization.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,"
          "clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_power_cap")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--once", action="store_true")
ap.add_argument("--interval-ms", type=int, default=500)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
cmd = ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,nounits"]
if not a.once:
    cmd += [f"--loop-ms={a.interval_ms}"]
with open(a.out, "w", newline="") as f:
    sys.exit(subprocess.call(cmd, stdout=f))
