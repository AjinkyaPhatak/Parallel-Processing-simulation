"""Turn raw per-repetition CSVs (results/raw) into processed statistics (results/processed).

Everything reported in the technical note is computed here from the raw data.
Definitions (also stated in the report):
  * time statistic           : median kernel time over timed repetitions (mean, std, CV, 95% CI also kept)
  * 95% CI                   : Student-t interval on the mean, mean +- t(0.975, n-1) * std / sqrt(n)
  * vecadd bytes / FLOPs     : 12 n bytes (2 reads + 1 write of float32), n FLOPs
  * matmul FLOPs / bytes     : 2 N^3 FLOPs; compulsory bytes 12 N^2 (A, B read once, C written once)
  * conv KxK FLOPs / bytes   : 2 K^2 N^2 FLOPs; compulsory bytes 8 N^2 (input read once, output written once)
  * efficiency               : achieved throughput / hardware peak (DRAM GB/s or FP32 GFLOP/s)
  * speedup_vs_cpu           : median single-thread CPU time / median GPU kernel time (kernel only)
  * speedup_vs_default       : median time of default config (256 threads 1D, 16x16 2D) / median time of config
  * rel_to_best              : best median time at that size / median time of config  (1.0 = optimum)
Usage: python scripts/analyze.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "results" / "raw", ROOT / "results" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
dev = json.loads((RAW / "device_info.json").read_text())
PEAK_BW = dev["peak_dram_bw_gbs"]

KEYS = ["kernel", "variant", "size", "bx", "by", "threads", "flag"]


def ci95(x):
    n = len(x)
    if n < 2:
        return np.nan
    return stats.t.ppf(0.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n)


def work(kernel, size):
    """(bytes, flops) per launch."""
    n = float(size)
    if kernel == "vecadd":
        return 12 * n, n
    if kernel == "matmul":
        return 12 * n * n, 2 * n ** 3
    if kernel.startswith("conv"):
        k = int(kernel[4:])
        return 8 * n * n, 2 * k * k * n * n
    raise ValueError(kernel)


def label(r):
    return f"{int(r.bx)}" if r.kernel == "vecadd" else f"{int(r.bx)}x{int(r.by)}"


# ---------------------------------------------------------------- GPU clocks during sweep
def clock_summary():
    p = RAW / "gpu_log_sweep.csv"
    if not p.exists():
        return None, pd.DataFrame()
    g = pd.read_csv(p, skipinitialspace=True)
    g.columns = [c.split(" [")[0].strip() for c in g.columns]
    g["t"] = pd.to_datetime(g["timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    busy = g[g["utilization.gpu"] > 50]
    st = pd.read_csv(RAW / "suite_times.csv", encoding="utf-8-sig")
    rows = []
    for _, s in st.iterrows():
        w = g[(g.t >= pd.Timestamp(s.start)) & (g.t <= pd.Timestamp(s.end))]
        wb = w[w["utilization.gpu"] > 50]
        rows.append(dict(suite=s.suite, seconds=s.seconds, samples=len(w), busy_samples=len(wb),
                         sm_clock_median_mhz=wb["clocks.current.sm"].median() if len(wb) else np.nan,
                         sm_clock_min_mhz=wb["clocks.current.sm"].min() if len(wb) else np.nan,
                         sm_clock_max_mhz=wb["clocks.current.sm"].max() if len(wb) else np.nan,
                         mem_clock_median_mhz=wb["clocks.current.memory"].median() if len(wb) else np.nan,
                         temp_max_c=w["temperature.gpu"].max(), power_max_w=w["power.draw"].max(),
                         pstates=";".join(sorted(w["pstate"].astype(str).unique())),
                         thermal_slowdown_samples=int((w["clocks_event_reasons.sw_thermal_slowdown"].astype(str).str.strip() == "Active").sum()),
                         hw_slowdown_samples=int((w["clocks_event_reasons.hw_slowdown"].astype(str).str.strip() == "Active").sum()),
                         power_cap_samples=int((w["clocks_event_reasons.sw_power_cap"].astype(str).str.strip() == "Active").sum())))
    cs = pd.DataFrame(rows)
    overall = busy["clocks.current.sm"].median() if len(busy) else np.nan
    cs.to_csv(OUT / "gpu_clock_summary.csv", index=False)
    return overall, cs


busy_clock, _ = clock_summary()
PEAK_FP32_REPORTED = dev["peak_fp32_gflops"]
PEAK_FP32_OBSERVED = (2 * dev["fp32_lanes_per_sm"] * dev["sm_count"] * busy_clock * 1e6 / 1e9
                      if busy_clock and not np.isnan(busy_clock) else np.nan)
PEAK_FP32 = PEAK_FP32_OBSERVED if not np.isnan(PEAK_FP32_OBSERVED) else PEAK_FP32_REPORTED
pd.DataFrame([dict(peak_dram_bw_gbs=PEAK_BW, peak_fp32_gflops_at_reported_clock=PEAK_FP32_REPORTED,
                   reported_sm_clock_mhz=dev["sm_clock_khz"] / 1e3, median_busy_sm_clock_mhz=busy_clock,
                   peak_fp32_gflops_at_observed_clock=PEAK_FP32_OBSERVED, peak_fp32_used=PEAK_FP32)]
             ).to_csv(OUT / "peaks_used.csv", index=False)

# ---------------------------------------------------------------- kernel statistics
kt = pd.read_csv(RAW / "kernel_times.csv")
cfg = pd.read_csv(RAW / "configs.csv")
agg = kt.groupby(KEYS).time_ms.agg(n="count", mean="mean", median="median", std="std", min="min", max="max").reset_index()
agg["cv"] = agg["std"] / agg["mean"]
agg["ci95_ms"] = kt.groupby(KEYS).time_ms.apply(lambda s: ci95(s.values)).values
agg["q1"] = kt.groupby(KEYS).time_ms.quantile(0.25).values
agg["q3"] = kt.groupby(KEYS).time_ms.quantile(0.75).values
agg["iqr_over_median"] = (agg.q3 - agg.q1) / agg["median"]


def median_ci(x):
    """Distribution-free 95% CI of the median (binomial order statistics)."""
    x = np.sort(x); n = len(x)
    lo = int(np.floor(n / 2 - 1.96 * np.sqrt(n) / 2)); hi = int(np.ceil(n / 2 + 1.96 * np.sqrt(n) / 2))
    return x[max(lo, 0)], x[min(hi, n - 1)]


mci = kt.groupby(KEYS).time_ms.apply(lambda s: median_ci(s.values))
agg["median_ci_lo"] = [v[0] for v in mci.values]
agg["median_ci_hi"] = [v[1] for v in mci.values]
agg["n_outliers_1p5x"] = kt.assign(m=kt.groupby(KEYS).time_ms.transform("median")).pipe(
    lambda d: (d.time_ms > 1.5 * d.m).groupby([d[k] for k in KEYS]).sum()).values
agg = agg.merge(cfg.drop(columns=["threads", "flag"]), on=["kernel", "variant", "size", "bx", "by"], how="left")
bw = agg.apply(lambda r: work(r.kernel, r["size"]), axis=1, result_type="expand")
agg["bytes"], agg["flops"] = bw[0], bw[1]
agg["gbs"] = agg["bytes"] / (agg["median"] * 1e-3) / 1e9
agg["gflops"] = agg["flops"] / (agg["median"] * 1e-3) / 1e9
agg["pct_peak_bw"] = 100 * agg["gbs"] / PEAK_BW
agg["pct_peak_fp32"] = 100 * agg["gflops"] / PEAK_FP32
agg["arith_intensity"] = agg["flops"] / agg["bytes"]
agg["config"] = agg.apply(label, axis=1)

# speedup vs CPU
cpu = pd.read_csv(RAW / "cpu_baseline.csv")
cpu_s = cpu.groupby(["kernel", "size"]).time_ms.agg(cpu_n="count", cpu_median_ms="median", cpu_min_ms="min",
                                                    cpu_max_ms="max").reset_index()
cpu_s.to_csv(OUT / "cpu_baseline_summary.csv", index=False)
agg = agg.merge(cpu_s[["kernel", "size", "cpu_median_ms"]], on=["kernel", "size"], how="left")
agg["speedup_vs_cpu"] = agg["cpu_median_ms"] / agg["median"]


# speedup vs default GPU config
def is_default(r):
    return (r.kernel == "vecadd" and r.bx == 256) or (r.kernel != "vecadd" and r.bx == 16 and r.by == 16)


dflt = agg[agg.apply(is_default, axis=1)][["kernel", "variant", "size", "median"]].rename(columns={"median": "default_median_ms"})
agg = agg.merge(dflt, on=["kernel", "variant", "size"], how="left")
agg["speedup_vs_default"] = agg["default_median_ms"] / agg["median"]

# relative to best per (kernel, variant, size), excluding nothing (flagged configs are still real configs)
best = agg.groupby(["kernel", "variant", "size"])["median"].transform("min")
agg["rel_to_best"] = best / agg["median"]
agg.to_csv(OUT / "kernel_stats.csv", index=False)

# ---------------------------------------------------------------- best configs + robustness
naive = agg[agg.variant == "naive"]
rows = []
for (k, s), g in naive.groupby(["kernel", "size"]):
    g = g.sort_values("median")
    b = g.iloc[0]
    within = g[g.rel_to_best >= 1 / 1.05]
    rows.append(dict(kernel=k, size=s, best_config=b.config, best_median_ms=b["median"], best_ci95_ms=b.ci95_ms,
                     best_median_ci_lo=b.median_ci_lo, best_median_ci_hi=b.median_ci_hi, best_cv=b.cv,
                     best_gbs=b.gbs, best_gflops=b.gflops, best_pct_peak_bw=b.pct_peak_bw,
                     best_pct_peak_fp32=b.pct_peak_fp32, best_theoretical_occ=b.theoretical_occupancy,
                     default_median_ms=b.default_median_ms, best_speedup_vs_default=b.speedup_vs_default,
                     best_speedup_vs_cpu=b.speedup_vs_cpu, worst_config=g.iloc[-1].config,
                     worst_over_best=g.iloc[-1]["median"] / b["median"],
                     n_within_5pct=len(within), configs_within_5pct=" ".join(within.config)))
bestdf = pd.DataFrame(rows)
bestdf.to_csv(OUT / "best_configs.csv", index=False)

rob = (naive.groupby(["kernel", "config", "bx", "by", "threads", "flag"])
       .agg(n_sizes=("size", "nunique"), worst_rel=("rel_to_best", "min"), mean_rel=("rel_to_best", "mean"),
            geomean_rel=("rel_to_best", lambda x: float(np.exp(np.mean(np.log(x))))),
            sizes_within_5pct=("rel_to_best", lambda x: int((x >= 1 / 1.05).sum())))
       .reset_index().sort_values(["kernel", "worst_rel", "mean_rel"], ascending=[True, False, False]))
rob["rank_in_kernel"] = rob.groupby("kernel").cumcount() + 1
rob.to_csv(OUT / "robustness.csv", index=False)
most_robust = rob[rob.rank_in_kernel == 1].set_index("kernel")

# ---------------------------------------------------------------- summary table
summ = bestdf.copy()
summ["most_robust_config"] = summ.kernel.map(most_robust.config)
summ["robust_worst_rel_pct"] = 100 * summ.kernel.map(most_robust.worst_rel)
summ["throughput"] = np.where(summ.kernel == "matmul", summ.best_gflops.round(1).astype(str) + " GFLOP/s",
                              summ.best_gbs.round(1).astype(str) + " GB/s")
summ["pct_peak"] = np.where(summ.kernel == "matmul", summ.best_pct_peak_fp32, summ.best_pct_peak_bw).round(1)
summ.to_csv(OUT / "summary_table.csv", index=False)
md = ["| Kernel | Size | Best config | Median time, ms [95% CI] | Throughput | % of peak | Speedup vs 16x16/256 | Speedup vs CPU | Most robust config (worst-case % of best) |",
      "|---|---|---|---|---|---|---|---|---|"]
for _, r in summ.iterrows():
    sz = f"2^{int(np.log2(r['size']))}" if r.kernel == "vecadd" else f"{int(r['size'])}²"
    md.append(f"| {r.kernel} | {sz} | {r.best_config} | {r.best_median_ms:.4g} [{r.best_median_ci_lo:.4g}, {r.best_median_ci_hi:.4g}] | {r.throughput} | "
              f"{r.pct_peak}% | {r.best_speedup_vs_default:.2f}× | {r.best_speedup_vs_cpu:.0f}× | "
              f"{r.most_robust_config} ({r.robust_worst_rel_pct:.1f}%) |")
(OUT / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- occupancy vs performance
occ_rows = []
for (k, s), g in naive.groupby(["kernel", "size"]):
    g = g[g.flag == 0]
    if g.theoretical_occupancy.nunique() > 1:
        pr, pp = stats.pearsonr(g.theoretical_occupancy, g.rel_to_best)
        sr, sp = stats.spearmanr(g.theoretical_occupancy, g.rel_to_best)
    else:
        pr = pp = sr = sp = np.nan
    occ_rows.append(dict(kernel=k, size=s, occupancy_source="theoretical", n=len(g), pearson_r=pr, pearson_p=pp,
                         spearman_rho=sr, spearman_p=sp))
ncu_p = OUT / "ncu_metrics.csv"
if ncu_p.exists():
    nc = pd.read_csv(ncu_p)
    m = nc.merge(naive, on=["kernel", "variant", "size", "bx", "by"], how="inner")
    for (k, s), g in m[m.flag == 0].groupby(["kernel", "size"]):
        if g.achieved_occupancy.nunique() > 1:
            pr, pp = stats.pearsonr(g.achieved_occupancy, g.rel_to_best)
            sr, sp = stats.spearmanr(g.achieved_occupancy, g.rel_to_best)
            occ_rows.append(dict(kernel=k, size=s, occupancy_source="achieved_ncu", n=len(g), pearson_r=pr,
                                 pearson_p=pp, spearman_rho=sr, spearman_p=sp))
pd.DataFrame(occ_rows).to_csv(OUT / "occupancy_correlation.csv", index=False)

# ncu metrics joined with timing results, for explaining the observed ranking
if ncu_p.exists():
    ex = nc.merge(agg[["kernel", "variant", "size", "bx", "by", "config", "flag", "median", "rel_to_best", "gbs",
                       "gflops", "bytes", "theoretical_occupancy"]], on=["kernel", "variant", "size", "bx", "by"], how="left")
    ex["dram_traffic_over_compulsory"] = (ex.dram_bytes_read + ex.dram_bytes_write) / ex["bytes"]
    keep = ["kernel", "variant", "size", "config", "flag", "median", "rel_to_best", "theoretical_occupancy",
            "achieved_occupancy", "occ_limit_blocks", "occ_limit_registers", "occ_limit_warps", "waves_per_sm_ncu",
            "sm_throughput_pct", "dram_throughput_pct", "l1tex_throughput_pct", "l1_hit_pct", "l2_hit_pct",
            "sectors_per_request", "ld_bytes_per_sector_pct", "dram_traffic_over_compulsory"] + \
           [c for c in ex.columns if c.startswith("stall_")]
    ex[keep].sort_values(["kernel", "variant", "size", "rel_to_best"], ascending=[True, True, True, False]) \
        .to_csv(OUT / "ncu_explained.csv", index=False)

# ---------------------------------------------------------------- data movement + breakdown
tr = pd.read_csv(RAW / "transfers.csv")
trs = tr.groupby(["kernel", "size", "direction", "mem_type", "bytes"]).time_ms.agg(
    n="count", median="median", std="std").reset_index()
trs["gbs"] = trs["bytes"] / (trs["median"] * 1e-3) / 1e9
trs.to_csv(OUT / "transfers_summary.csv", index=False)
bd = []
for _, r in bestdf.iterrows():
    for mem in ["pageable", "pinned"]:
        t = trs[(trs.kernel == r.kernel) & (trs["size"] == r["size"]) & (trs.mem_type == mem)].set_index("direction")["median"]
        h2d, d2h = t.get("H2D", np.nan), t.get("D2H", np.nan)
        for which, kt_ms in [("best", r.best_median_ms), ("default", r.default_median_ms)]:
            tot = h2d + kt_ms + d2h
            bd.append(dict(kernel=r.kernel, size=r["size"], mem_type=mem, kernel_config=which, h2d_ms=h2d,
                           kernel_ms=kt_ms, d2h_ms=d2h, total_ms=tot, kernel_fraction=kt_ms / tot,
                           transfer_fraction=(h2d + d2h) / tot,
                           speedup_vs_cpu_incl_transfers=cpu_s.set_index(["kernel", "size"]).loc[(r.kernel, r["size"]), "cpu_median_ms"] / tot))
pd.DataFrame(bd).to_csv(OUT / "time_breakdown.csv", index=False)

# ---------------------------------------------------------------- launch / sync overhead
ov_p = RAW / "launch_overhead.csv"
if ov_p.exists():
    ov = pd.read_csv(ov_p)
    ovs = ov.groupby(["test", "threads", "blocks"]).time_us_per_launch.agg(
        n="count", median="median", mean="mean", std="std", min="min", max="max").reset_index()
    ovs.to_csv(OUT / "overhead_summary.csv", index=False)

# ---------------------------------------------------------------- drift / noise checks
dr = []
for key, g in kt.groupby(KEYS):
    g = g.sort_values("rep")
    rho, p = stats.spearmanr(g.rep, g.time_ms) if g.time_ms.nunique() > 1 else (np.nan, np.nan)
    h = len(g) // 2
    dr.append(dict(zip(KEYS, key), n=len(g), spearman_rep_vs_time=rho, p=p,
                   first_half_median=g.time_ms.iloc[:h].median(), second_half_median=g.time_ms.iloc[h:].median(),
                   half_change_pct=100 * (g.time_ms.iloc[h:].median() / g.time_ms.iloc[:h].median() - 1)))
drift = pd.DataFrame(dr)
drift.to_csv(OUT / "drift_check.csv", index=False)

# Sensitivity: do conclusions change if only the second half of repetitions (steady state) is used?
sens = []
ss = kt[kt.rep >= kt.groupby(["kernel", "variant", "size"]).rep.transform("max") // 2 + 1]
ssm = ss.groupby(KEYS).time_ms.median().rename("median_ss").reset_index()
both = agg[KEYS + ["median", "config"]].merge(ssm, on=KEYS)
both = both[both.variant == "naive"]
both["rel_ss"] = both.groupby(["kernel", "size"]).median_ss.transform("min") / both.median_ss
for (k, s), g in both.groupby(["kernel", "size"]):
    rho = stats.spearmanr(g["median"], g.median_ss)[0]
    sens.append(dict(kernel=k, size=s, spearman_rank_full_vs_steady=rho,
                     best_full=g.loc[g["median"].idxmin(), "config"], best_steady=g.loc[g.median_ss.idxmin(), "config"],
                     median_shift_pct=100 * (g.median_ss / g["median"] - 1).median()))
sens = pd.DataFrame(sens)
rob_ss = both.groupby(["kernel", "config"]).rel_ss.min().reset_index().sort_values(["kernel", "rel_ss"], ascending=[True, False])
top_ss = rob_ss.groupby("kernel").head(1).set_index("kernel").config
sens["most_robust_full"] = sens.kernel.map(most_robust.config)
sens["most_robust_steady"] = sens.kernel.map(top_ss)
sens.to_csv(OUT / "sensitivity_steady_state.csv", index=False)

print(f"peak BW {PEAK_BW:.1f} GB/s, peak FP32 used {PEAK_FP32:.0f} GFLOP/s (busy SM clock median {busy_clock} MHz)")
print(f"configs: {len(agg)}, failed correctness: {(cfg['pass'] == 0).sum()}")
print(f"CV: median {agg.cv.median():.3f}, max {agg.cv.max():.3f}; configs with CV>5%: {(agg.cv > 0.05).sum()}")
print(f"drift: |first->second half| median change >2%: {(drift.half_change_pct.abs() > 2).sum()} of {len(drift)}")
print(f"IQR/median: median {agg.iqr_over_median.median():.3f}; rows >1.5x median: {int(agg.n_outliers_1p5x.sum())} of {len(kt)}")
print(sens.round(3).to_string(index=False))
print((OUT / "summary_table.md").read_text(encoding="utf-8"))
print(rob.groupby("kernel").head(3)[["kernel", "config", "worst_rel", "mean_rel", "sizes_within_5pct"]].to_string(index=False))
print(pd.read_csv(OUT / "occupancy_correlation.csv").round(3).to_string(index=False))

# ---------------------------------------------------------------- compact summary table used in the note
bdp = pd.read_csv(OUT / "time_breakdown.csv")
bdp = bdp[(bdp.kernel_config == "best") & (bdp.mem_type == "pageable")].set_index(["kernel", "size"])
L = ["| Kernel | Size | Best config | Median ms [95% CI] | Throughput (% of peak) | vs. default | vs. CPU (kernel / end-to-end) |",
     "|---|---|---|---|---|---|---|"]
for _, r in summ.iterrows():
    sz = f"2^{int(np.log2(r['size']))}" if r.kernel == "vecadd" else f"{int(r['size'])}²"
    e2e = bdp.loc[(r.kernel, r["size"]), "speedup_vs_cpu_incl_transfers"]
    L.append(f"| {r.kernel} | {sz} | {r.best_config} | {r.best_median_ms:.4g} [{r.best_median_ci_lo:.4g}, "
             f"{r.best_median_ci_hi:.4g}] | {r.throughput} ({r.pct_peak}%) | {r.best_speedup_vs_default:.2f}× | "
             f"{r.best_speedup_vs_cpu:.0f}× / {e2e:.3g}× |")
(OUT / "summary_table_compact.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- cross-kernel robustness (2D kernels)
x2 = naive[naive.kernel != "vecadd"]
xk = (x2.groupby(["config", "threads", "flag"])
      .agg(worst_rel_all=("rel_to_best", "min"), mean_rel_all=("rel_to_best", "mean"),
           worst_matmul=("rel_to_best", lambda s: s[x2.loc[s.index, "kernel"] == "matmul"].min()),
           worst_conv3=("rel_to_best", lambda s: s[x2.loc[s.index, "kernel"] == "conv3"].min()),
           worst_conv5=("rel_to_best", lambda s: s[x2.loc[s.index, "kernel"] == "conv5"].min()))
      .reset_index().sort_values("worst_rel_all", ascending=False))
xk.to_csv(OUT / "robustness_cross_kernel_2d.csv", index=False)
print(xk.head(6).round(3).to_string(index=False))
