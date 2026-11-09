# RVC-Train-GGML dev build script (Windows / PowerShell 5.1+)
#
# NOTE: keep this file ASCII-only. PowerShell 5.1 mis-parses UTF-8 files without BOM;
#       documentation lives in docs/ instead.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1              # configure + build
#   powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1 -Action configure
#   powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1 -Action build
#   powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1 -Action run  # run both exes selftest
#   powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1 -Action clean
#
# Extra CMake args: -CmakeArgs "-DRVCTRAIN_TRAINCPP_DIR=D:\path\to\train.cpp"
# Toolchain overrides via env: RVCT_VCVARS / RVCT_CMAKE / RVCT_NINJA

param(
    [ValidateSet('all', 'configure', 'build', 'run', 'clean')]
    [string]$Action = 'all',
    [string]$CmakeArgs = ''
)

$ErrorActionPreference = 'Stop'

$Root     = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $Root 'build'

$VcVars = $env:RVCT_VCVARS
if (-not $VcVars) {
    # locate vcvars64.bat via vswhere (any VS 2017+ edition)
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsRoot) { $VcVars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat' }
    }
}
if (-not $VcVars) {
    throw "vcvars64.bat not found (install Visual Studio C++ tools, or set RVCT_VCVARS)"
}

$CMake = $env:RVCT_CMAKE
if (-not $CMake) {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmd) { $CMake = $cmd.Source }
}
$Ninja = $env:RVCT_NINJA
if (-not $Ninja) {
    $cmd = Get-Command ninja -ErrorAction SilentlyContinue
    if ($cmd) { $Ninja = $cmd.Source }
}
# Fall back to the CMake/Ninja bundled with Visual Studio (not always on PATH).
if (-not $CMake -and $vsRoot) {
    $candidate = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (Test-Path -LiteralPath $candidate) { $CMake = $candidate }
}
if (-not $Ninja -and $vsRoot) {
    $candidate = Join-Path $vsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
    if (Test-Path -LiteralPath $candidate) { $Ninja = $candidate }
}

foreach ($tool in @($VcVars)) {
    if (-not $tool -or -not (Test-Path -LiteralPath $tool)) {
        throw "tool not found: $tool (override via RVCT_VCVARS)"
    }
}
# cmake / ninja may be a bare command name resolvable on PATH
foreach ($pair in @(@('RVCT_CMAKE', $CMake), @('RVCT_NINJA', $Ninja))) {
    $tool = $pair[1]
    $ok = $tool -and ((Test-Path -LiteralPath $tool) -or (Get-Command $tool -ErrorAction SilentlyContinue))
    if (-not $ok) {
        throw "tool not found: $tool (override via $($pair[0]))"
    }
}

function Invoke-Vs([string]$CmdLine) {
    Write-Host ">>> $CmdLine"
    cmd /c "call `"$VcVars`" >nul 2>&1 && $CmdLine"
    if ($LASTEXITCODE -ne 0) {
        throw "command failed (exit $LASTEXITCODE): $CmdLine"
    }
}

if ($Action -eq 'clean') {
    Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue
    Write-Host 'build dir removed'
    exit 0
}

if ($Action -in @('all', 'configure')) {
    $configCmd = "`"$CMake`" -S `"$Root`" -B `"$BuildDir`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$Ninja`" -DCMAKE_BUILD_TYPE=Release $CmakeArgs"
    Invoke-Vs $configCmd
}

if ($Action -in @('all', 'build')) {
    Invoke-Vs "`"$CMake`" --build `"$BuildDir`""
}

if ($Action -eq 'run') {
    $prep  = Join-Path $BuildDir 'bin\rvc-prep.exe'
    $train = Join-Path $BuildDir 'bin\rvc-train.exe'
    Write-Host '--- rvc-prep --list-devices ---'
    & $prep --list-devices
    Write-Host '--- rvc-train --list-devices ---'
    & $train --list-devices
    Write-Host '--- rvc-prep selftest ---'
    & $prep --selftest
    $prepOk = ($LASTEXITCODE -eq 0)
    Write-Host '--- rvc-train selftest ---'
    & $train --selftest
    $trainOk = ($LASTEXITCODE -eq 0)
    if (-not ($prepOk -and $trainOk)) {
        throw "selftest failed (rvc-prep=$prepOk, rvc-train=$trainOk)"
    }
}

Write-Host "done: $Action"
