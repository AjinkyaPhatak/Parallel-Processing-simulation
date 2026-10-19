# Full timed sweep. Writes raw per-repetition CSVs to results\raw and a GPU clock/temp/power log.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_sweep.ps1
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Raw = "results\raw"
New-Item -ItemType Directory -Force $Raw | Out-Null

# Only the sweep outputs are cleared (device_info.json / idle snapshot are kept).
foreach ($f in @("kernel_times.csv", "configs.csv", "transfers.csv", "cpu_baseline.csv", "launch_overhead.csv",
                 "gpu_log_sweep.csv", "sweep_log.txt", "suite_times.csv")) {
    if (Test-Path "$Raw\$f") { Remove-Item "$Raw\$f" -Confirm:$false }
}

# Repetition plan (warm-up = 5 discarded launches per configuration everywhere).
$Suites = @(
    @{ name = "vecadd";       reps = 50 },
    @{ name = "conv3";        reps = 50 },
    @{ name = "conv5";        reps = 50 },
    @{ name = "matmul_tiled"; reps = 50 },
    @{ name = "matmul";       reps = 30 },
    @{ name = "overhead";     reps = 50 }
)

nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.max.sm,clocks.max.mem --format=csv |
    Out-File -Encoding utf8 "$Raw\gpu_static.csv"
python scripts\gpu_monitor.py --once "$Raw\gpu_state_pre_sweep.csv"
$mon = Start-Process python -ArgumentList "scripts\gpu_monitor.py", "$Raw\gpu_log_sweep.csv", "--interval-ms", "500" -PassThru -WindowStyle Hidden

"suite,start,end,seconds" | Out-File -Encoding utf8 "$Raw\suite_times.csv"
try {
    foreach ($s in $Suites) {
        $t0 = Get-Date
        Write-Host "=== $($s.name) reps=$($s.reps)"
        & .\build\bench.exe $s.name --reps $s.reps --warmup 5 --outdir $Raw 2>&1 | Tee-Object -Append -FilePath "$Raw\sweep_log.txt"
        if ($LASTEXITCODE -ne 0) { throw "suite $($s.name) failed with exit code $LASTEXITCODE" }
        $t1 = Get-Date
        "$($s.name),$($t0.ToString('s')),$($t1.ToString('s')),$([int]($t1 - $t0).TotalSeconds)" |
            Out-File -Append -Encoding utf8 "$Raw\suite_times.csv"
    }
} finally {
    Stop-Process -Id $mon.Id -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='nvidia-smi.exe'" | Where-Object { $_.CommandLine -match "loop-ms" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
}
python scripts\gpu_monitor.py --once "$Raw\gpu_state_post_sweep.csv"
Write-Host "sweep complete"
