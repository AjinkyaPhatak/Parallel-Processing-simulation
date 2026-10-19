"""Parse Nsight Compute raw-page CSV exports (results/raw/ncu/*_raw.csv) into results/processed/ncu_metrics.csv.
One row per profiled launch = one (kernel, size, block config). Problem size is recovered from grid x block
extent (nearest of the sizes that were profiled)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NCU, OUT = ROOT / "results" / "raw" / "ncu", ROOT / "results" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
SIZES = {"vecadd": [1048576, 16777216], "matmul": [1024, 2048], "matmul_tiled": [1024, 2048],
         "conv3": [2048, 4096], "conv5": [2048, 4096]}
COLS = {
    "achieved_occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_ncu": "sm__maximum_warps_per_active_cycle_pct",
    "occ_limit_blocks": "launch__occupancy_limit_blocks",
    "occ_limit_registers": "launch__occupancy_limit_registers",
    "occ_limit_shared_mem": "launch__occupancy_limit_shared_mem",
    "occ_limit_warps": "launch__occupancy_limit_warps",
    "waves_per_sm_ncu": "launch__waves_per_multiprocessor",
    "regs_per_thread_ncu": "launch__registers_per_thread",
    "duration_ns_ncu": "gpu__time_duration.sum",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
    "ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "ld_bytes_per_sector_pct": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "st_bytes_per_sector_pct": "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
    "l1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
}
STALLS = ["long_scoreboard", "lg_throttle", "mio_throttle", "math_pipe_throttle", "not_selected", "barrier",
          "tex_throttle", "wait", "short_scoreboard"]
for s in STALLS:
    COLS[f"stall_{s}"] = f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio"


SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
         "ns": 1, "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6, "s": 1e9}


def num(v):
    if isinstance(v, str):
        v = v.replace(",", "").strip()
        if v in ("", "n/a"):
            return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


rows = []
for suite, sizes in SIZES.items():
    p = NCU / f"{suite}_raw.csv"
    if not p.exists():
        print("missing", p)
        continue
    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    units = df.iloc[0]
    df = df.iloc[1:]  # units row
    for _, r in df.iterrows():
        bx, by, _ = [int(x) for x in re.findall(r"\d+", str(r["Block Size"]))]
        gx, gy, _ = [int(x) for x in re.findall(r"\d+", str(r["Grid Size"]))]
        extent = gx * bx if suite == "vecadd" else gy * by
        size = min(sizes, key=lambda s: abs(s - extent))
        kernel = "matmul" if suite.startswith("matmul") else suite
        rec = dict(kernel=kernel, variant="tiled" if suite == "matmul_tiled" else "naive", size=size, bx=bx, by=by)
        for k, c in COLS.items():
            v = num(r[c]) if c in df.columns else np.nan
            u = str(units.get(c, "")) if c in df.columns else ""
            rec[k] = v * SCALE.get(u, 1.0)   # normalise byte -> bytes and time -> ns
        rec["kernel_name"] = r["Kernel Name"]
        rows.append(rec)
nc = pd.DataFrame(rows)
nc["sectors_per_request"] = nc.ld_sectors / nc.ld_requests
nc["achieved_occupancy"] = nc["achieved_occupancy"] / 100.0      # fraction, like theoretical_occupancy
nc["theoretical_occupancy_ncu"] = nc["theoretical_occupancy_ncu"] / 100.0
nc = nc.drop_duplicates(subset=["kernel", "variant", "size", "bx", "by"], keep="first")
nc.to_csv(OUT / "ncu_metrics.csv", index=False)
print(f"ncu rows: {len(nc)}  ->  {OUT / 'ncu_metrics.csv'}")
print(nc.groupby(["kernel", "variant", "size"]).size())
