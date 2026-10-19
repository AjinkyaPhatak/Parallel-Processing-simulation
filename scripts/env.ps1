# Activate the portable toolchain in ./tools for the current PowerShell session.
# Usage:  . .\scripts\env.ps1
$Root  = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root "tools"

$Cuda  = Join-Path $Tools "cuda"
$Ncu   = (Get-ChildItem (Join-Path $Tools "nsight-compute") -Recurse -Filter ncu.bat | Select-Object -First 1).DirectoryName
$Msvc  = (Get-ChildItem (Join-Path $Tools "msvc\VC\Tools\MSVC") -Directory | Sort-Object Name | Select-Object -Last 1).FullName
$Kits  = Join-Path $Tools "msvc\Windows Kits\10"
$SdkV  = (Get-ChildItem (Join-Path $Kits "Include") -Directory | Sort-Object Name | Select-Object -Last 1).Name

$env:CUDA_PATH = $Cuda
$env:PATH = @(
    (Join-Path $Cuda "bin"), $Ncu,
    (Join-Path $Msvc "bin\Hostx64\x64"),
    (Join-Path $Kits "bin\$SdkV\x64"),
    (Join-Path $Tools "pandoc"),
    $env:PATH) -join ";"
$env:INCLUDE = @(
    (Join-Path $Msvc "include"),
    (Join-Path $Kits "Include\$SdkV\ucrt"),
    (Join-Path $Kits "Include\$SdkV\shared"),
    (Join-Path $Kits "Include\$SdkV\um"),
    (Join-Path $Kits "Include\$SdkV\winrt")) -join ";"
$env:LIB = @(
    (Join-Path $Msvc "lib\x64"),
    (Join-Path $Kits "Lib\$SdkV\ucrt\x64"),
    (Join-Path $Kits "Lib\$SdkV\um\x64")) -join ";"
Write-Host "Toolchain active: MSVC $(Split-Path $Msvc -Leaf), Windows SDK $SdkV, CUDA $Cuda"
