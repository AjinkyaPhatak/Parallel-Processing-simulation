# One command to reproduce everything from scratch (after scripts\setup_toolchain.py has populated .\tools):
#   build -> device query -> timed sweep -> Nsight Compute subset -> analysis -> plots -> report.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 [-SkipNcu] [-SkipReport]
param([switch]$SkipNcu, [switch]$SkipReport)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
. "$PSScriptRoot\env.ps1"
$t0 = Get-Date

& powershell -ExecutionPolicy Bypass -File scripts\build.ps1;              if ($LASTEXITCODE) { throw "build failed" }
New-Item -ItemType Directory -Force results\raw | Out-Null
python scripts\gpu_monitor.py --once results\raw\gpu_state_idle.csv
@((Get-CimInstance Win32_Processor).Name, ((Get-CimInstance Win32_OperatingSystem).Caption + " " +
  (Get-CimInstance Win32_OperatingSystem).BuildNumber)) | Out-File -Encoding utf8 results\raw\host_info.txt
& .\build\device_query.exe results\raw\device_info.json;                  if ($LASTEXITCODE) { throw "device query failed" }
& powershell -ExecutionPolicy Bypass -File scripts\run_sweep.ps1;          if ($LASTEXITCODE) { throw "sweep failed" }
if (-not $SkipNcu) {
    & powershell -ExecutionPolicy Bypass -File scripts\profile_ncu.ps1;    if ($LASTEXITCODE) { throw "ncu failed" }
}
python scripts\analyze.py;                                                if ($LASTEXITCODE) { throw "analysis failed" }
python scripts\make_plots.py;                                             if ($LASTEXITCODE) { throw "plots failed" }
if (-not $SkipReport) { python scripts\build_report.py;                   if ($LASTEXITCODE) { throw "report build failed" } }
Write-Host "run_all finished in $([int]((Get-Date) - $t0).TotalMinutes) min"
