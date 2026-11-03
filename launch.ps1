# toaster-loop launcher (Windows) — double-click / terminal entry point.
#
#   .\launch.ps1              # start the dashboard (front+back end) + open browser
#   .\launch.ps1 --stop       # stop the dashboard this launcher started
#   .\launch.ps1 --no-browser
#
# PowerShell port of launch.sh (A3): idempotent — if a launched dashboard is
# already running, `toaster-loop launch` just re-opens its URL. Uses the repo venv;
# runs `toaster-loop launch`, which supervises everything and records state under
# TOASTER_LOOP_HOME (default %USERPROFILE%\.toaster-loop).
#
# Double-click friendly via launch.cmd, or run:
#   powershell -ExecutionPolicy Bypass -File launch.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Windows PowerShell 5.1 turns any text a native command writes to stderr into
# a NativeCommandError, which is terminating under $ErrorActionPreference =
# "Stop"; pwsh 7 does not do this. toaster-loop logs progress to stderr, so run it in
# a child scope pinned to Continue and rely on $LASTEXITCODE for failures.
function Native([scriptblock]$Body) {
    & { $ErrorActionPreference = "Continue"; . $Body }
}

$ToasterLoopBin = Join-Path $PSScriptRoot ".venv\Scripts\toaster-loop.exe"
if (-not (Test-Path $ToasterLoopBin)) {
    Write-Host "toaster-loop is not set up yet - running setup.ps1 first."
    & (Join-Path $PSScriptRoot "setup.ps1") --no-training
    if ($LASTEXITCODE -ne 0) {
        Write-Error "setup failed; see output above."
        exit 1
    }
}

# NOTE: `$args` inside the Native scriptblock would resolve to that block's own
# (empty) automatic variable, so capture the launcher arguments first.
$LaunchArgs = $args
Native { & $ToasterLoopBin launch @LaunchArgs }
exit $LASTEXITCODE
