@echo off
rem Activate the portable toolchain in .\tools for the current cmd session.  Usage: call scripts\env.bat
set "ROOT=%~dp0.."
set "TOOLS=%ROOT%\tools"
for /f "delims=" %%d in ('dir /b /ad /on "%TOOLS%\msvc\VC\Tools\MSVC"') do set "MSVCV=%%d"
for /f "delims=" %%d in ('dir /b /ad /on "%TOOLS%\msvc\Windows Kits\10\Include"') do set "SDKV=%%d"
for /f "delims=" %%f in ('dir /b /s "%TOOLS%\nsight-compute\ncu.bat"') do set "NCUDIR=%%~dpf"
set "MSVC=%TOOLS%\msvc\VC\Tools\MSVC\%MSVCV%"
set "KITS=%TOOLS%\msvc\Windows Kits\10"
set "CUDA_PATH=%TOOLS%\cuda"
set "PATH=%CUDA_PATH%\bin;%NCUDIR%;%MSVC%\bin\Hostx64\x64;%KITS%\bin\%SDKV%\x64;%TOOLS%\pandoc;%PATH%"
set "INCLUDE=%MSVC%\include;%KITS%\Include\%SDKV%\ucrt;%KITS%\Include\%SDKV%\shared;%KITS%\Include\%SDKV%\um;%KITS%\Include\%SDKV%\winrt"
set "LIB=%MSVC%\lib\x64;%KITS%\Lib\%SDKV%\ucrt\x64;%KITS%\Lib\%SDKV%\um\x64"
echo Toolchain active: MSVC %MSVCV%, Windows SDK %SDKV%, CUDA %CUDA_PATH%
