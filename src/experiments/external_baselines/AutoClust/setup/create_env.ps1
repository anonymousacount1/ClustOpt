<#
.SYNOPSIS
    AutoClust reuses the existing `ml2dac` conda env -- no separate env is created.
    Native Windows is discouraged for real runs (SMAC + fork-based hard timeout);
    use WSL. This script just runs the readiness check via the ml2dac env.
#>
[CmdletBinding()] param([string]$EnvName = "ml2dac")
$ErrorActionPreference = "Stop"
Write-Host "AutoClust reuses the '$EnvName' conda env (no new env)." -ForegroundColor Cyan
conda run -n $EnvName python experiments\external_baselines\AutoClust\setup\check_autoclust_install.py --repo-root (Get-Location).Path
Write-Host "[note] Run the final subfamily experiments under WSL (setup_wsl.md)." -ForegroundColor Yellow
