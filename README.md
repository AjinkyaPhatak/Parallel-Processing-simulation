# Case S02: CUDA Thread-Block Configuration

**Research question:** Which grid/block configuration gives robust performance across workload sizes?

This repository holds an individual contribution to a batch HPC case study. It sweeps block size **and block shape** for three kernels (vector add, naive SGEMM, and 3×3 / 5×5 convolution). For each configuration it measures time, throughput and occupancy (theoretical and achieved), and scores each configuration by its worst-case fraction of the per-size optimum.

Deliverables:

| What | Where |
|---|---|
| Technical note (3 pages) | `report/technical_note.md`, `.docx`, `.pdf` |
| Verified references, with how each was verified | `report/references_verified.md` |
| Source code | `src/` (CUDA) and `scripts/` (build, run, profile, analysis, plots, report) |
| Raw data, one row per repetition | `results/raw/*.csv` |
| Processed statistics | `results/processed/*.csv` |
| Plots | `plots/*.png` and `*.svg`; compact versions used in the note are in `plots/report/` |

## 1. Hardware and software used

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 laptop (Turing, CC 7.5, 30 SMs, 6 GB), WDDM driver 581.57, PCIe Gen3 ×8 |
| CPU / OS | AMD Ryzen 7 3750H, Windows 11 Pro (build 26200) |
| Toolchain | CUDA 12.9.1 (nvcc 12.9.86), Nsight Compute 2025.2.1, MSVC 19.44 (14.44), Windows SDK 10.0.26100, pandoc 3.12.1, Python 3.12 (pandas, numpy, scipy, matplotlib, python-docx, pypdf, pymupdf) |

All device numbers used in the analysis come from `results/raw/device_info.json`, which `build/device_query.exe` queries at run time.

## 2. Toolchain setup (portable, no admin rights)

The whole toolchain lives in `./tools/` and nothing is installed system-wide. `setup_toolchain.py` downloads the official NVIDIA redistributable archives and Microsoft's VS 2022 Build Tools packages, verifies their SHA-256 hashes, and unpacks them. Running it implies acceptance of the NVIDIA CUDA EULA and the Microsoft Build Tools / Windows SDK licence. It downloads about 1 GB, which takes roughly 30 minutes at 0.5–1 MB/s.

```bash
python scripts/setup_toolchain.py
```

```bash
python -m pip install pandas numpy scipy matplotlib python-docx pypdf pymupdf
```

Activate the toolchain in each new PowerShell session with `. .\scripts\env.ps1`, or in cmd with `scripts\env.bat`.

> **Quirk worth knowing.** nvcc rejects a portable MSVC with the error *"Host compiler targets unsupported OS"* unless `VC\Auxiliary\Build\vcvarsall.bat` and `vcvars64.bat` exist. `setup_toolchain.py` creates small stand-ins at that path that call `scripts\env.bat`.
>
> **Disk space.** After setup, `tools/_downloads/` (about 0.9 GB) can be deleted.

## 3. One-command reproduction

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

`run_all` runs these steps in order:

1. Build
2. Device query and host info
3. Timed sweep, about 14 min (834 s here)
4. Nsight Compute subset, about 6 min (375 s here)
5. Analysis
6. Plots
7. Report

Use `-SkipNcu` or `-SkipReport` to leave out a step.

### Individual steps

| Step | Command | Output |
|---|---|---|
| Build | `powershell -ExecutionPolicy Bypass -File scripts\build.ps1` | `build\device_query.exe`, `build\bench.exe` (`-O3 -arch=sm_75 -lineinfo -std=c++17`; host `/O2 /fp:precise`) |
| Device query | `build\device_query.exe results\raw\device_info.json` | SM count, limits, clocks, derived peaks |
| Timed sweep | `powershell -ExecutionPolicy Bypass -File scripts\run_sweep.ps1` | `results\raw\{kernel_times,configs,transfers,cpu_baseline,launch_overhead,gpu_log_sweep,suite_times}.csv` |
| Profile | `powershell -ExecutionPolicy Bypass -File scripts\profile_ncu.ps1` | `results\raw\ncu\*.ncu-rep`, `*_raw.csv`, `ncu_commands.txt` → `results\processed\ncu_metrics.csv` |
| Analysis | `python scripts\analyze.py` | `results\processed\*.csv`, `summary_table*.md` |
| Plots | `python scripts\make_plots.py` | `plots\fig1…fig9.{png,svg}`, `plots\report\r1…r5.{png,svg}` |
| Report | `python scripts\build_report.py` | `report\technical_note.docx` / `.pdf`, and the page count (it must print `pages: 3`) |

A single benchmark run looks like this:

```bash
build\bench.exe matmul --sizes 1024,2048 --reps 30 --warmup 5 --outdir results\raw
```

The available suites are `vecadd`, `matmul`, `matmul_tiled`, `conv3`, `conv5` and `overhead`. Options: `--reps`, `--warmup`, `--cpu-reps`, `--seed`, `--preheat <s>`.

## 4. Experimental protocol, in brief

- **Kernels** (`src/kernels.cuh`). The naive kernels are the study's baseline. The tiled SGEMM (tile = block = 8, 16 or 32) is used only as a secondary comparison.
- **Sizes.**
  - Vector add: 2^20, 2^24, 2^27
  - Matmul: N = 512, 1024, 2048, 4096
  - Convolution: 1024², 2048², 4096²
- **Configurations.**
  - 1D: 32, 64, 96, 128, 192, 256, 384, 512, 768 and 1024 threads, plus the flagged non-multiples 48 and 100.
  - 2D: 24 shapes grouped by thread count (64, 128, 256, 512 and 1024), plus the flagged 12×12.
  - `threadIdx.x` maps to the contiguous (column) dimension.
- **Correctness.** Before a configuration is timed, its output buffer is filled with NaN, the kernel runs once, and the result is compared with a single-thread CPU reference. Any configuration that fails is excluded and logged. In this run, all 288 configurations passed (`configs.csv`, columns `max_abs_err`, `max_rel_err`, `pass`).
- **Timing.**
  - Each configuration gets 5 discarded warm-up launches.
  - Then 50 timed repetitions follow (30 for naive matmul), timed with cudaEvent (kernel only).
  - The **configuration order is reshuffled every repetition** with a fixed seed.
  - A 2 s GPU pre-heat runs before each sweep.
  - nvidia-smi logs SM and memory clocks, temperature, power and throttle reasons every 500 ms.
- **Separately timed costs.**
  - H2D/D2H copies, both pageable and pinned (`transfers.csv`).
  - Empty-kernel launch cost, the block-scheduling cost of an empty 2^24-thread grid, the host-side cost of the launch call, launch+sync round trips, and idle `cudaDeviceSynchronize` (`launch_overhead.csv`).
- **Statistics** (`analyze.py`).
  - Per configuration: median, mean, std, min, max, CV, IQR, a Student-t 95% CI of the mean, and a distribution-free 95% CI of the median.
  - Checks on the data: drift (first half vs second half of the repetitions), a steady-state sensitivity re-analysis, and a count of outliers above 1.5× the median.

### Metric definitions (also stated in the note)

- **Bytes.** Compulsory bytes: vector add 12·n, matmul 12·N², convolution 8·N².
- **FLOPs.** Vector add n, matmul 2·N³, convolution 2·K²·N².
- **Efficiency.** Achieved throughput divided by peak.
  - Peak DRAM bandwidth = 2 × memory clock × bus width / 8, from the queried values: 264 GB/s.
  - Peak FP32 = 2 × 64 lanes × SMs × the median SM clock observed under load (`peaks_used.csv`).
- **Speedup.**
  - (a) Against the single-thread CPU baseline (MSVC `/O2`; matmul uses i-k-j loop order), both kernel-only and end-to-end including copies.
  - (b) Against the default GPU configuration: 256 threads (1D) or 16×16 (2D).
- **Robustness.** For each configuration, `rel_to_best` = (best median at that size) / (this configuration's median). The configuration's robustness is the worst case (and mean) of `rel_to_best` across sizes. `robustness_cross_kernel_2d.csv` also takes the worst case across all 2D kernels.

### Exact Nsight Compute command lines

`profile_ncu.ps1` writes the command lines it actually ran to `results/raw/ncu/ncu_commands.txt`. Here is one of them, with the metric list abbreviated:

```text
ncu --target-processes all --kernel-name matmul_naive_kernel \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,
              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,
              smsp__sass_average_data_bytes_per_sector_mem_global_op_{ld,st}.pct,l1tex__t_sector_hit_rate.pct,
              lts__t_sector_hit_rate.pct,smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
    --section SpeedOfLight --section Occupancy --section LaunchStats --force-overwrite -o results\raw\ncu\matmul
    build\bench.exe matmul --sizes 1024,2048 --warmup 0 --reps 0 --preheat 0 --cpu-reps 1 --outdir results\raw\ncu\_bench_out
ncu --import results\raw\ncu\matmul.ncu-rep --csv --page raw > results\raw\ncu\matmul_raw.csv
```

The profiled subset covers every configuration at two sizes per kernel: vector add 2^20 and 2^24, matmul and tiled matmul 1024 and 2048, convolution 2048² and 4096². That is 174 kernel launches, each profiled in 11 replay passes. ncu's defaults were kept: `--clock-control base` and `--cache-control all`. Its metrics are therefore used to *explain* the results, never as timings.

## 5. Data dictionary (main files)

| File | One row per | Key columns |
|---|---|---|
| `raw/kernel_times.csv` | timed repetition | kernel, variant, size, bx, by, threads, flag, rep, order_pos, time_ms, timestamp |
| `raw/configs.csv` | (kernel, size, config) | grid, num_blocks, regs_per_thread, max_active_blocks_per_sm, theoretical_occupancy, waves_per_sm, max_abs_err, max_rel_err, pass |
| `raw/transfers.csv` | copy repetition | direction, mem_type, bytes, time_ms |
| `raw/cpu_baseline.csv` | CPU run | kernel, size, time_ms |
| `raw/launch_overhead.csv` | overhead repetition | test, threads, blocks, time_us_per_launch |
| `raw/gpu_log_sweep.csv` | 500 ms sample | P-state, temperature, power, SM/memory clock, throttle reasons |
| `processed/kernel_stats.csv` | (kernel, size, config) | all statistics, GB/s, GFLOP/s, % of peak, speedups, rel_to_best |
| `processed/robustness*.csv` | config | worst_rel, mean_rel, sizes_within_5pct |
| `processed/ncu_explained.csv` | profiled launch | achieved occupancy, occupancy limits, L1/L2 hit rates, sectors/request, bytes/sector %, DRAM traffic ÷ compulsory bytes, stall reasons |
| `processed/summary_table*.{csv,md}` | (kernel, size) | best config, time with CI, throughput, % of peak, speedups, most robust config |

## 6. What could not be done, and why

- **Clock locking** (`nvidia-smi -lgc`) needs administrator rights and changes a system setting, so it was not used. The laptop GPU hit its power cap during the long matmul sweep, which is why randomized interleaving, clock logging and a steady-state sensitivity check were used instead. See the limitations section of the note.
- **WSL/Linux comparison.** There is no WSL distribution on the machine, so every result comes from Windows WDDM. Its launch overhead (≈25 µs) is higher than on Linux or TCC drivers.
- **Separating synchronisation cost inside a kernel.** `__syncthreads` appears only in the tiled kernel. Nsight Compute reports its barrier stalls, but its cost cannot be cleanly separated from compute time. Launch, synchronisation and block-scheduling overheads *were* measured, in `launch_overhead.csv`.
- **Ryoo et al. (PPoPP 2008).** This was a suggested reference, but it was not verified in this session, so it was not cited. The nine cited works were all verified (see `report/references_verified.md`).
- **The Nsight Compute subset.** Profiling every size was not affordable, so two sizes per kernel were profiled (vector add 2^27 and matmul 4096 are not profiled).

## 7. Folder layout

```
src/            device_query.cu, kernels.cuh, bench.cu
scripts/        setup_toolchain.py, env.ps1/.bat, build.ps1, run_sweep.ps1, profile_ncu.ps1, gpu_monitor.py,
                parse_ncu.py, analyze.py, make_plots.py, build_report.py, run_all.ps1
results/raw/    raw CSVs, device_info.json, GPU logs, ncu reports (results/raw/ncu/)
results/processed/  statistics, summary tables, robustness, occupancy correlation, ncu joins
plots/          fig1…fig9 (PNG+SVG), plots/report/ compact figures for the note
report/         technical_note.{md,docx,pdf}, references_verified.md
tools/          portable toolchain (not part of the submission; recreate with setup_toolchain.py)
build/          compiled executables
```
