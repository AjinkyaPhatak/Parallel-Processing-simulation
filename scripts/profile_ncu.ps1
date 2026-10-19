# Nsight Compute profiling of a representative subset: ALL block configurations, two problem sizes per kernel.
# Each configuration is launched exactly once under the profiler (--warmup 0 --reps 0 --preheat 0: only the
# correctness-check launch runs), so one profiled launch = one configuration.
# ncu defaults kept on purpose: --clock-control base (clocks locked to base -> stable counters, but durations are
# NOT comparable with the cudaEvent timings) and --cache-control all (caches flushed between replay passes).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\profile_ncu.ps1
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Out = "results\raw\ncu"
New-Item -ItemType Directory -Force $Out | Out-Null
$Tmp = "results\raw\ncu\_bench_out"   # bench's own CSVs from profiled runs are kept apart from the timed sweep
New-Item -ItemType Directory -Force $Tmp | Out-Null

$Metrics = @(
  "sm__warps_active.avg.pct_of_peak_sustained_active",
  "dram__bytes_read.sum", "dram__bytes_write.sum",
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
  "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
  "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
  "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio"
) -join ","
$Sections = @("--section", "SpeedOfLight", "--section", "Occupancy", "--section", "LaunchStats")

$Runs = @(
  @{ suite = "vecadd";       kernel = "vecadd_kernel";       sizes = "1048576,16777216" },
  @{ suite = "matmul";       kernel = "matmul_naive_kernel"; sizes = "1024,2048" },
  @{ suite = "matmul_tiled"; kernel = "regex:matmul_tiled";  sizes = "1024,2048" },
  @{ suite = "conv3";        kernel = "regex:conv2d_kernel"; sizes = "2048,4096" },
  @{ suite = "conv5";        kernel = "regex:conv2d_kernel"; sizes = "2048,4096" }
)
$cmdLog = "$Out\ncu_commands.txt"
"# exact ncu command lines used ($(Get-Date -Format s))" | Out-File -Encoding utf8 $cmdLog
foreach ($r in $Runs) {
    $rep = "$Out\$($r.suite)"
    $args_ = @("--target-processes", "all", "--kernel-name", $r.kernel, "--metrics", $Metrics) + $Sections +
             @("--force-overwrite", "-o", $rep, "build\bench.exe", $r.suite, "--sizes", $r.sizes,
               "--warmup", "0", "--reps", "0", "--preheat", "0", "--cpu-reps", "1", "--outdir", $Tmp)
    "ncu " + ($args_ -join " ") | Out-File -Append -Encoding utf8 $cmdLog
    Write-Host "=== ncu $($r.suite)"
    & ncu.bat @args_
    if ($LASTEXITCODE -ne 0) { throw "ncu failed for $($r.suite)" }
    & ncu.bat --import "$rep.ncu-rep" --csv --page raw | Out-File -Encoding utf8 "$Out\$($r.suite)_raw.csv"
}
python scripts\parse_ncu.py
