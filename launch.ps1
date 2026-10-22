# sable launcher (Windows) — double-click / terminal entry point.
#
#   .\launch.ps1              # start the dashboard (front+back end) + open browser
#   .\launch.ps1 --stop       # stop the dashboard this launcher started
#   .\launch.ps1 --no-browser
#
# PowerShell port of launch.sh (A3): idempotent — if a launched dashboard is
# already running, `sable launch` just re-opens its URL. Uses the repo venv;
# runs `sable launch`, which supervises everything and records state under
# SABLE_HOME (default %USERPROFILE%\.sable).
#
# Double-click friendly via launch.cmd, or run:
#   powershell -ExecutionPolicy Bypass -File launch.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$sable = Join-Path $PSScriptRoot ".venv\Scripts\sable.exe"
if (-not (Test-Path $sable)) {
    Write-Host "sable is not set up yet - running setup.ps1 first."
    & (Join-Path $PSScriptRoot "setup.ps1") --no-training
    if ($LASTEXITCODE -ne 0) {
        Write-Error "setup failed; see output above."
        exit 1
    }
}

& $sable launch @args
exit $LASTEXITCODE
