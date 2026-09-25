<#
.SYNOPSIS
    Create a dedicated, isolated Windows venv for ML2DAC.
.DESCRIPTION
    ML2DAC needs Python 3.9 + a legacy stack (smac 1.4.0, hdbscan, pymfe, ...).
    Native Windows is DISCOURAGED (smac/pyrfr + hdbscan need SWIG + MSVC build
    tools, and several pins lack Windows wheels). Prefer WSL (see setup_wsl.md).
    This creates `.venv_ml2dac` at the repo root; run
    install_ml2dac_requirements.ps1 afterwards.
.PARAMETER PythonLauncher
    How to invoke Python 3.9, e.g. "py -3.9". Default "py -3.9".
#>
[CmdletBinding()]
param([string]$PythonLauncher = "py -3.9", [string]$EnvName = ".venv_ml2dac")
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$envPath  = Join-Path $repoRoot $EnvName
Write-Host "Repo root : $repoRoot"
Write-Host "Env path  : $envPath"
Write-Host "Python    : $PythonLauncher"

if (Test-Path $envPath) {
    Write-Host "[info] $EnvName already exists." -ForegroundColor Yellow
} else {
    $parts = $PythonLauncher.Split(" "); $exe = $parts[0]
    $rest = @(); if ($parts.Length -gt 1) { $rest = $parts[1..($parts.Length-1)] }
    & $exe @rest -m venv $envPath
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed with '$PythonLauncher'. Use WSL (setup_wsl.md)." }
}
$venvPy = Join-Path $envPath "Scripts\python.exe"
& $venvPy -m pip install --upgrade pip setuptools wheel
& $venvPy --version
Write-Host "[done] $EnvName ready. Next: install_ml2dac_requirements.ps1 (env activated)." -ForegroundColor Green
Write-Host "If smac/hdbscan fail to build, switch to WSL (setup_wsl.md) -- strongly recommended."
