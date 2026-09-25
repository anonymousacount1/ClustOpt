<#
.SYNOPSIS
    Install a curated AutoML4Clust dependency stack into the ACTIVE environment.

.DESCRIPTION
    Installs only the packages the AutoML4Clust SMAC clustering path actually
    needs. The upstream requirements.txt pins many Ubuntu-system packages
    (systemd-python, python-apt, ufw, cloud-init, ...) that cannot install on
    Windows and are unused by AutoML4Clust; this script installs the curated
    subset instead.

    Run this AFTER activating the dedicated env created by create_env.ps1:
        .\.venv_automl4clust\Scripts\Activate.ps1

.NOTES
    * smac==0.12.0 pulls pyrfr, a SWIG/C++ extension. Install SWIG + MSVC C++
      Build Tools first (see setup_windows.md, step 3) or this will fail.
    * If SMAC cannot be built on Windows, use WSL (setup_wsl.md) -- by far the
      most reliable path for this legacy stack.
#>
[CmdletBinding()]
param(
    [switch]$SkipSmac,        # install everything except smac (debug the rest first)
    [switch]$IncludePymfe     # meta-learning/warmstart only; not needed for the SMAC baseline
)

$ErrorActionPreference = "Stop"

# Use the active environment's python.
$py = (Get-Command python).Source
Write-Host "Using python: $py"
& $py --version

function Pip-Install([string[]]$pkgs) {
    Write-Host "[pip] $($pkgs -join ' ')" -ForegroundColor Cyan
    & $py -m pip install @pkgs
    if ($LASTEXITCODE -ne 0) { throw "pip install failed for: $($pkgs -join ' ')" }
}

# 1) Core numerical stack compatible with the legacy scikit-learn.
Pip-Install @("Cython==0.29.17")
Pip-Install @("numpy==1.18.4", "scipy==1.4.1", "pandas==1.0.3")

# 2) Legacy scikit-learn + extras used for k-Medoids and clustering metrics.
Pip-Install @("scikit-learn==0.21.3")
Pip-Install @("scikit-learn-extra==0.0.3")
Pip-Install @("pyclustering")

# 3) Configuration space + optimizers.
Pip-Install @("ConfigSpace==0.4.12")
Pip-Install @("scikit-optimize==0.5.2")
Pip-Install @("Pyro4", "serpent")            # hpbandster runtime deps
Pip-Install @("hpbandster==0.7.4")

# 4) SMAC (Bayesian optimizer) -- the primary optimizer for the baseline.
if ($SkipSmac) {
    Write-Host "[info] Skipping smac (per -SkipSmac). The 'Random' optimizer fallback can be used meanwhile." -ForegroundColor Yellow
} else {
    Pip-Install @("smac==0.12.0")
    # smac 0.12 calls pynisher.enforce_limits, removed in pynisher 1.x; pip pulls
    # 1.x by default and SMAC crashes on the first run -- pin the 0.5.0 API.
    Pip-Install @("pynisher==0.5.0")
}

# 5) Optional meta-learning (NOT used by the SMAC baseline / warmstarting is off).
if ($IncludePymfe) {
    Pip-Install @("pymfe==0.0.3")
}

Write-Host ""
Write-Host "[done] Curated AutoML4Clust stack installed." -ForegroundColor Green
Write-Host "Verify: python -m experiments.external_baselines.AutoML4Clust.runners.smoke_test_automl4clust"
