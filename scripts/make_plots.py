"""Regenerate every figure from the processed CSVs.  Usage:  python scripts/analyze.py && python scripts/make_plots.py
Outputs PNG (200 dpi) + SVG for each figure into plots/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC, RAW, PLOTS = ROOT / "results" / "processed", ROOT / "results" / "raw", ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# validated categorical order (fixed, never cycled) + single-hue sequential ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "lines.markersize": 4.5, "figure.dpi": 100, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    "svg.fonttype": "none"})

ks = pd.read_csv(PROC / "kernel_stats.csv")
cfg = pd.read_csv(RAW / "configs.csv")
dev = json.loads((RAW / "device_info.json").read_text())
peaks = pd.read_csv(PROC / "peaks_used.csv").iloc[0]
ncu = pd.read_csv(PROC / "ncu_metrics.csv") if (PROC / "ncu_metrics.csv").exists() else None
KERNELS = ["vecadd", "matmul", "conv3", "conv5"]
TITLES = {"vecadd": "Vector add (1D)", "matmul": "Naive matmul (2D)", "conv3": "Conv 3×3 (2D)", "conv5": "Conv 5×5 (2D)"}


def size_label(k, s):
    return f"n = 2^{int(np.log2(s))}" if k == "vecadd" else f"N = {int(s)}"


def config_order(k):
    c = cfg[(cfg.kernel == k) & (cfg.variant == "naive")].drop_duplicates(["bx", "by"])
    c = c.assign(lab=[f"{b}" if k == "vecadd" else f"{b}x{y}" for b, y in zip(c.bx, c.by)])
    return c.sort_values(["threads", "flag"], kind="stable") if k == "vecadd" else c.sort_values(["threads", "flag"], kind="stable")


def flagged_labels(ax, order):
    for t, (_, r) in zip(ax.get_xticklabels(), order.iterrows()):
        if r.flag:
            t.set_color(SERIES[7]); t.set_fontstyle("italic")


def save(fig, name):
    fig.savefig(PLOTS / f"{name}.png", dpi=200)
    fig.savefig(PLOTS / f"{name}.svg")
    plt.close(fig)
    print("wrote", name)


def per_config_panel(ax, k, ycol, err=None, ylog=True):
    d = ks[(ks.kernel == k) & (ks.variant == "naive")]
    order = config_order(k)
    x = np.arange(len(order))
    for i, s in enumerate(sorted(d["size"].unique())):
        g = d[d["size"] == s].set_index(["bx", "by"]).reindex(list(zip(order.bx, order.by)))
        yerr = (np.vstack([g[ycol].values - g["median_ci_lo"].values, g["median_ci_hi"].values - g[ycol].values])
                if err == "median_ci" else None)
        ax.errorbar(x, g[ycol].values, yerr=yerr, color=SERIES[i], marker="o", ms=plt.rcParams["lines.markersize"], lw=plt.rcParams["lines.linewidth"], capsize=1.5,
                    elinewidth=0.8, label=size_label(k, s))
    ax.set_xticks(x)
    ax.set_xticklabels(order.lab, rotation=90)
    flagged_labels(ax, order)
    if ylog:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 3.5)   # headroom for the legend
    else:
        lo, hi = ax.get_ylim(); ax.set_ylim(0, hi * 1.18)
    ax.set_title(TITLES[k])
    return order


# ---------------------------------------------------------------- Fig 1: kernel time vs block config
def fig1(figsize=(11, 7.2), name="fig1_time_vs_block", title=True):
  fig, axs = plt.subplots(2, 2, figsize=figsize)
  for ax, k in zip(axs.flat, KERNELS):
    per_config_panel(ax, k, "median", err="median_ci")
    ax.set_ylabel("Kernel time, median (ms, log)")
    ax.set_xlabel("Threads per block" if k == "vecadd" else "Block shape bx × by (grouped by threads/block)")
    ax.legend(frameon=False, ncol=4, loc="upper center")
  if title:
    fig.suptitle("Kernel time vs block configuration (error bars: distribution-free 95% CI of the median; red italic = not a multiple of 32)", fontsize=10)
  fig.tight_layout()
  save(fig, name)


fig1()

# ---------------------------------------------------------------- Fig 2: occupancy vs block config
def fig2(figsize=(11, 7.2), name="fig2_occupancy_vs_block", title=True):
  fig, axs = plt.subplots(2, 2, figsize=figsize)
  for ax, k in zip(axs.flat, KERNELS):
    order = config_order(k)
    x = np.arange(len(order))
    th = cfg[(cfg.kernel == k) & (cfg.variant == "naive")].drop_duplicates(["bx", "by"]).set_index(["bx", "by"])
    th = th.reindex(list(zip(order.bx, order.by)))
    ax.plot(x, 100 * th.theoretical_occupancy.values, color=INK2, lw=plt.rcParams["lines.linewidth"], marker="s", ms=plt.rcParams["lines.markersize"],
            label="theoretical (CUDA occupancy API)")
    if ncu is not None:
        n = ncu[(ncu.kernel == k) & (ncu.variant == "naive")]
        for i, s in enumerate(sorted(n["size"].unique())):
            g = n[n["size"] == s].set_index(["bx", "by"]).reindex(list(zip(order.bx, order.by)))
            ax.plot(x, 100 * g.achieved_occupancy.values, color=SERIES[i], marker="o", ms=plt.rcParams["lines.markersize"],
                    label=f"achieved (ncu), {size_label(k, s)}")
    ax.set_ylim(0, 118)   # ncu's sampled achieved occupancy can slightly exceed 100% on short kernels
    ax.set_xticks(x); ax.set_xticklabels(order.lab, rotation=90); flagged_labels(ax, order)
    ax.set_ylabel("Occupancy (% of 32 warps/SM)")
    ax.set_title(TITLES[k] + f"  ({int(th.regs_per_thread.iloc[0])} regs/thread)")
    ax.legend(frameon=False, loc="lower right")
  if title:
    fig.suptitle("Theoretical and achieved occupancy vs block configuration", fontsize=10)
  fig.tight_layout()
  save(fig, name)


fig2()

# ---------------------------------------------------------------- Fig 3: throughput vs block config
def fig3(figsize=(11, 7.2), name="fig3_throughput_vs_block", title=True):
  fig, axs = plt.subplots(2, 2, figsize=figsize)
  for ax, k in zip(axs.flat, KERNELS):
    col = "gflops" if k == "matmul" else "gbs"
    per_config_panel(ax, k, col, ylog=(k == "matmul"))
    if k == "matmul":
        ax.set_ylabel("Throughput (GFLOP/s, log)")
    else:
        ax.axhline(peaks.peak_dram_bw_gbs, color=INK2, lw=0.9)
        ax.text(ax.get_xlim()[1], peaks.peak_dram_bw_gbs * 0.985, f"peak DRAM {peaks.peak_dram_bw_gbs:.0f} GB/s",
                color=INK2, fontsize=plt.rcParams["legend.fontsize"], va="top", ha="right")
        ax.set_ylim(0, peaks.peak_dram_bw_gbs * 1.25)
        ax.set_ylabel("Effective bandwidth (GB/s)")
    ax.legend(frameon=False, ncol=4, loc="upper center")
  if title:
    fig.suptitle("Throughput vs block configuration", fontsize=10)
  fig.tight_layout()
  save(fig, name)


fig3()

# ---------------------------------------------------------------- Fig 4: block-shape heatmaps (largest size)
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])  # dark = best
for ax, k in zip(axs, ["matmul", "conv3", "conv5"]):
    d = ks[(ks.kernel == k) & (ks.variant == "naive")]
    s = d["size"].max(); d = d[d["size"] == s]
    bxs, bys = sorted(d.bx.unique()), sorted(d.by.unique())
    M = np.full((len(bys), len(bxs)), np.nan)
    for _, r in d.iterrows():
        M[bys.index(r.by), bxs.index(r.bx)] = 100 * r.rel_to_best
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, origin="lower", aspect="auto")
    for i in range(len(bys)):
        for j in range(len(bxs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 55 else INK)
    ax.set_xticks(range(len(bxs))); ax.set_xticklabels(bxs)
    ax.set_yticks(range(len(bys))); ax.set_yticklabels(bys)
    ax.set_xlabel("blockDim.x (contiguous direction)"); ax.set_ylabel("blockDim.y")
    ax.set_title(f"{TITLES[k]}, N = {int(s)}"); ax.grid(False)
fig.colorbar(im, ax=axs, shrink=0.9, label="% of best (best / config median time)")
fig.suptitle("Effect of block SHAPE: performance relative to the best configuration (blank = shape not tested)", fontsize=10)
save(fig, "fig4_shape_heatmap")

# ---------------------------------------------------------------- Fig 5: robustness across sizes
rob = pd.read_csv(PROC / "robustness.csv")


def fig5(figsize=(11, 7.2), name="fig5_robustness", title=True):
  fig, axs = plt.subplots(2, 2, figsize=figsize)
  for ax, k in zip(axs.flat, KERNELS):
    order = config_order(k)
    x = np.arange(len(order))
    d = ks[(ks.kernel == k) & (ks.variant == "naive")]
    for i, s in enumerate(sorted(d["size"].unique())):
        g = d[d["size"] == s].set_index(["bx", "by"]).reindex(list(zip(order.bx, order.by)))
        ax.plot(x, 100 * g.rel_to_best.values, ls="none", marker="o", ms=plt.rcParams["lines.markersize"] + 0.5, color=SERIES[i], label=size_label(k, s))
    r = rob[rob.kernel == k].set_index(["bx", "by"]).reindex(list(zip(order.bx, order.by)))
    ax.bar(x, 100 * r.worst_rel.values, color=GRID, width=0.7, zorder=0, label="worst case across sizes")
    best = rob[(rob.kernel == k) & (rob.rank_in_kernel == 1)].iloc[0]
    ax.axhline(95, color=INK2, lw=0.8)
    ax.text(len(order) - 0.5, 96, "95% of best", ha="right", fontsize=plt.rcParams["legend.fontsize"], color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(order.lab, rotation=90); flagged_labels(ax, order)
    ax.set_ylim(0, 105); ax.set_ylabel("% of per-size best")
    ax.set_title(f"{TITLES[k]} — most robust: {best.config} (worst {100 * best.worst_rel:.1f}%)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
  if title:
    fig.suptitle("Robustness: performance of each configuration relative to the optimum at each problem size", fontsize=10)
  fig.tight_layout()
  save(fig, name)


fig5()

# ---------------------------------------------------------------- Fig 6: occupancy vs performance scatter
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
for ax, k in zip(axs, KERNELS):
    d = ks[(ks.kernel == k) & (ks.variant == "naive") & (ks.flag == 0)]
    if ncu is not None and len(ncu[(ncu.kernel == k) & (ncu.variant == "naive")]):
        m = ncu[(ncu.kernel == k) & (ncu.variant == "naive")].merge(d, on=["kernel", "variant", "size", "bx", "by"])
        for i, s in enumerate(sorted(m["size"].unique())):
            g = m[m["size"] == s]
            r = np.corrcoef(g.achieved_occupancy, g.rel_to_best)[0, 1]
            ax.scatter(100 * g.achieved_occupancy, 100 * g.rel_to_best, s=22, color=SERIES[i],
                       edgecolor="white", linewidth=0.8, label=f"achieved, {size_label(k, s)} (r={r:.2f})")
        ax.set_xlabel("Achieved occupancy, ncu (%)")
    else:
        for i, s in enumerate(sorted(d["size"].unique())[:3]):
            g = d[d["size"] == s]
            ax.scatter(100 * g.theoretical_occupancy, 100 * g.rel_to_best, s=22, color=SERIES[i],
                       edgecolor="white", linewidth=0.8, label=size_label(k, s))
        ax.set_xlabel("Theoretical occupancy (%)")
    ax.set_title(TITLES[k]); ax.legend(frameon=False, fontsize=7, loc="lower left")
axs[0].set_ylabel("% of per-size best performance")
fig.suptitle("Does higher occupancy mean higher performance? (multiple-of-32 configs only)", fontsize=10)
fig.tight_layout()
save(fig, "fig6_occupancy_vs_performance")

# ---------------------------------------------------------------- Fig 7: H2D / kernel / D2H breakdown
bd = pd.read_csv(PROC / "time_breakdown.csv")
bd = bd[(bd.mem_type == "pageable") & (bd.kernel_config == "best")].copy()
bd["lab"] = [f"{k}\n{size_label(k, s).split('= ')[1]}" for k, s in zip(bd.kernel, bd["size"])]
fig, ax = plt.subplots(figsize=(11, 3.8))
x = np.arange(len(bd))
bottom = np.zeros(len(bd))
for col, c, name in [("h2d_ms", SERIES[0], "H2D copy"), ("kernel_ms", SERIES[1], "kernel (best config)"),
                     ("d2h_ms", SERIES[2], "D2H copy")]:
    frac = 100 * bd[col].values / bd.total_ms.values
    ax.bar(x, frac, bottom=bottom, color=c, width=0.72, label=name, edgecolor="white", linewidth=1.2)
    bottom += frac
for xi, t in zip(x, bd.total_ms.values):
    ax.text(xi, 101.5, f"{t:.3g} ms", ha="center", fontsize=7, color=INK2, rotation=90, va="bottom")
ax.set_xticks(x); ax.set_xticklabels(bd.lab, fontsize=7.5)
ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("% of end-to-end time")
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.16))
ax.set_title("Data movement vs computation (pageable host memory; label = total ms)", pad=22)
save(fig, "fig7_time_breakdown")

# ---------------------------------------------------------------- Fig 8: scalability
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, k in zip(axs, KERNELS):
    d = ks[(ks.kernel == k) & (ks.variant == "naive")]
    r = rob[rob.kernel == k]
    pick = [("most robust", r.iloc[0].config), ("default", "256" if k == "vecadd" else "16x16"),
            ("least robust", r.iloc[-1].config)]
    col = "gflops" if k == "matmul" else "gbs"
    seen = set()
    for i, (name, c) in enumerate(pick):
        if c is None or c in seen:
            continue
        seen.add(c)
        g = d[d.config == c].sort_values("size")
        ax.plot(g["size"], g[col], marker="o", color=SERIES[i], label=f"{c} ({name})")
    if k == "matmul":
        t = ks[(ks.kernel == k) & (ks.variant == "tiled")]
        for j, c in enumerate(sorted(t.config.unique())):
            g = t[t.config == c].sort_values("size")
            ax.plot(g["size"], g[col], marker="^", lw=1, color=SERIES[3 + j],
                    label=f"tiled {c} (secondary)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Problem size (elements)" if k == "vecadd" else "N (matrix/image side)")
    ax.set_ylabel("GFLOP/s" if k == "matmul" else "GB/s")
    ax.set_title(TITLES[k]); ax.legend(frameon=False, fontsize=6.5)
fig.suptitle("Scalability: throughput vs problem size for selected configurations", fontsize=10)
fig.tight_layout()
save(fig, "fig8_scalability")

# ---------------------------------------------------------------- Fig 9: launch / scheduling overhead
ov_p = PROC / "overhead_summary.csv"
if ov_p.exists():
    ov = pd.read_csv(ov_p)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    a = ov[ov.test == "empty_1block_device"].sort_values("threads")
    b = ov[ov.test == "empty_grid_2^24_threads"].sort_values("threads")
    axs[0].plot(a.threads, a["median"], marker="o", color=SERIES[0], label="1 block, back-to-back launches")
    axs[0].set_xscale("log", base=2); axs[0].set_ylim(0, a["median"].max() * 1.3)
    axs[0].set_xlabel("Threads per block"); axs[0].set_ylabel("µs per launch (median)")
    axs[0].set_title("Per-launch cost of an empty kernel"); axs[0].legend(frameon=False)
    axs[1].plot(b.threads, b["median"], marker="o", color=SERIES[1], label="median (incl. low-clock states)")
    axs[1].plot(b.threads, b["min"], marker="s", color=SERIES[0], label="minimum (full-clock floor)")
    axs[1].set_xscale("log", base=2); axs[1].set_ylim(0, b["median"].max() * 1.15)
    axs[1].set_xlabel("Threads per block (blocks = 2^24 / threads)"); axs[1].set_ylabel("µs per launch (median)")
    axs[1].set_title("Empty grid of 2^24 threads: cost grows with block count"); axs[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig9_launch_overhead")


# ---------------------------------------------------------------- compact versions for the 2-3 page note
(PLOTS / "report").mkdir(exist_ok=True)
with plt.rc_context({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 5.8,
                     "xtick.labelsize": 5.2, "ytick.labelsize": 5.8, "lines.linewidth": 1.0, "lines.markersize": 2.5}):
    fig1((6.9, 3.05), "report/r1_time_vs_block", title=False)
    fig2((6.9, 3.05), "report/r2_occupancy_vs_block", title=False)
    fig3((6.9, 3.05), "report/r3_throughput_vs_block", title=False)
    fig5((6.9, 3.6), "report/r5_robustness_per_kernel", title=False)

    # r4: the research-question answer -- worst case over sizes (1D) and over kernels x sizes (2D)
    xk = pd.read_csv(PROC / "robustness_cross_kernel_2d.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [1, 2.3]})
    rv = rob[rob.kernel == "vecadd"].sort_values("threads")
    xv = np.arange(len(rv))
    a1.bar(xv, 100 * rv.worst_rel, color=[SERIES[7] if f else SERIES[0] for f in rv.flag], width=0.7)
    a1.set_xticks(xv); a1.set_xticklabels(rv.config, rotation=90)
    a1.set_ylim(0, 128); a1.set_yticks([0, 20, 40, 60, 80, 100]); a1.set_ylabel("worst-case % of best"); a1.set_title("Vector add: worst over 3 sizes")
    a1.axhline(95, color=INK2, lw=0.7)
    x = np.arange(len(xk))
    a2.bar(x, 100 * xk.worst_rel_all, color=[SERIES[7] if f else SERIES[0] for f in xk.flag], width=0.7,
           label="worst over all 2D kernels and sizes")
    for j, (col, nm) in enumerate([("worst_matmul", "matmul"), ("worst_conv3", "conv3x3"), ("worst_conv5", "conv5x5")]):
        a2.plot(x, 100 * xk[col], ls="none", marker="o", ms=2.6, color=SERIES[j + 1], label=f"worst, {nm}")
    a2.set_xticks(x); a2.set_xticklabels(xk.config, rotation=90)
    a2.set_ylim(0, 128); a2.set_yticks([0, 20, 40, 60, 80, 100]); a2.axhline(95, color=INK2, lw=0.7)
    a2.set_title("2D kernels: ranked by worst case over matmul, conv3x3, conv5x5 and all sizes")
    a2.legend(frameon=False, ncol=4, loc="upper center", handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "report/r4_cross_kernel_robustness")
