<#
.SYNOPSIS
    AutoClust needs NO additional packages beyond the ML2DAC stack. Reuse the
    ml2dac env; if missing, install the ML2DAC requirements first.
#>
[CmdletBinding()] param()
Write-Host "AutoClust reuses the ml2dac env -- no extra requirements." -ForegroundColor Cyan
Write-Host "If the ml2dac env is missing, run the ML2DAC installer:" -ForegroundColor Yellow
Write-Host "  experiments\external_baselines\ML2DAC\setup\install_ml2dac_requirements.ps1"
