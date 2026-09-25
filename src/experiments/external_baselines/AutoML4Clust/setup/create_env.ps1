<#
.SYNOPSIS
    Create a dedicated, isolated Windows virtual environment for AutoML4Clust.

.DESCRIPTION
    AutoML4Clust requires a legacy dependency stack that must NOT be installed
    into the main project .venv. This script creates a separate venv named
    `.venv_automl4clust` at the repo root (git-ignored) and upgrades pip
    tooling. It does NOT install the AutoML4Clust requirements -- run
    install_automl4clust_requirements.ps1 afterwards.

.PARAMETER PythonLauncher
    How to invoke the target Python, e.g. "py -3.7" (recommended) or a full path
    to a python.exe. Defaults to "py -3.7".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File create_env.ps1 -PythonLauncher "py -3.7"
#>
[CmdletBinding()]
param(
    [string]$PythonLauncher = "py -3.7",
    [string]$EnvName = ".venv_automl4clust"
)

$ErrorActionPreference = "Stop"

# Resolve repo root: this script lives at
#   <repo>/experiments/external_baselines/AutoML4Clust/setup/create_env.ps1
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$envPath  = Join-Path $repoRoot $EnvName

Write-Host "Repo root : $repoRoot"
Write-Host "Env path  : $envPath"
Write-Host "Python    : $PythonLauncher"

if (Test-Path $envPath) {
    Write-Host "[info] $EnvName already exists. Reusing it (delete it to recreate)." -ForegroundColor Yellow
} else {
    Write-Host "[step] Creating virtual environment..."
    # Split the launcher so 'py -3.7' is invoked correctly.
    $parts = $PythonLauncher.Split(" ")
    $exe   = $parts[0]
    $rest  = @()
    if ($parts.Length -gt 1) { $rest = $parts[1..($parts.Length-1)] }
    & $exe @rest -m venv $envPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create venv with '$PythonLauncher'. Try a different Python (py -3.7 / py -3.8)."
    }
}

$venvPy = Join-Path $envPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv python not found at $venvPy" }

Write-Host "[step] Upgrading pip / setuptools / wheel..."
& $venvPy -m pip install --upgrade "pip<24" "setuptools<60" "wheel"

& $venvPy --version
Write-Host ""
Write-Host "[done] Dedicated AutoML4Clust env ready at $envPath" -ForegroundColor Green
Write-Host "Next:"
Write-Host "  1) Activate : .\$EnvName\Scripts\Activate.ps1"
Write-Host "  2) Install  : powershell -ExecutionPolicy Bypass -File experiments\external_baselines\AutoML4Clust\setup\install_automl4clust_requirements.ps1"
Write-Host "  3) Verify   : python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust"
