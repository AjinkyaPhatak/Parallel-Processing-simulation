# Build all executables into .\build using the portable toolchain.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\build.ps1
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force build | Out-Null
# -O3 (device + host), -arch=sm_75 (Turing), -lineinfo (source correlation in Nsight Compute).
# Host side: MSVC has no /O3; /O2 is its maximum-speed level. /fp:precise keeps CPU float semantics IEEE.
$Flags = @("-O3", "-arch=sm_75", "-lineinfo", "-std=c++17", "-Xcompiler", "/O2,/fp:precise,/EHsc,/nologo")
foreach ($t in @("device_query", "bench")) {
    Write-Host "nvcc $Flags -o build\$t.exe src\$t.cu"
    & nvcc @Flags -o "build\$t.exe" "src\$t.cu"
    if ($LASTEXITCODE -ne 0) { throw "build of $t failed" }
}
Write-Host "build OK"
