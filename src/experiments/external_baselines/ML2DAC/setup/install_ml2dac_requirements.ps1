<#
.SYNOPSIS
    Install the ML2DAC stack into the ACTIVE Windows environment.
.DESCRIPTION
    Native Windows is discouraged: smac==1.4.0 (pyrfr) and hdbscan==0.8.32 need
    SWIG + MSVC C++ Build Tools, and not all pins have Windows wheels. If this
    fails, use WSL (setup_wsl.md / create_env_wsl_conda.sh) -- the reliable path.
#>
[CmdletBinding()] param()
$ErrorActionPreference = "Stop"
$py = (Get-Command python).Source
& $py --version
function Pin([string[]]$pkgs) { Write-Host "[pip] $($pkgs -join ' ')" -ForegroundColor Cyan; & $py -m pip install @pkgs; if ($LASTEXITCODE -ne 0) { throw "pip failed: $($pkgs -join ' ')" } }

Pin @("numpy==1.24.2","scipy==1.8.0")
Pin @("ConfigSpace==0.6.0")
Pin @("smac==1.4.0")
Pin @("hdbscan==0.8.32")
Pin @("scikit-learn==0.24.2","joblib==1.0.0")
Pin @("pymfe==0.4.1")
try { Pin @("Definitions==0.2.0") } catch { Write-Host "Definitions optional; skipped" -ForegroundColor Yellow }
# Pin pandas LAST: pymfe pulls pandas 2.x, which removes SettingWithCopyWarning
# (imported by ML2DAC ApplicationPhase). Re-pin 1.1.5 so the import works.
Pin @("pandas==1.1.5")
Write-Host "[done] Verify: python experiments\external_baselines\ML2DAC\setup\check_ml2dac_install.py" -ForegroundColor Green
