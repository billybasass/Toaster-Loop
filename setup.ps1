# sable one-click setup (Windows) — safe to double-click or run from a terminal:
#
#   .\setup.ps1                          # full setup (venv + package + training
#                                        #  extras + llama.cpp build for this GPU)
#   .\setup.ps1 --no-training            # skip the heavy torch training stack
#   .\setup.ps1 --backend cuda|rocm|vulkan|cpu   # override GPU autodetect
#
# PowerShell port of setup.sh (A3). What it does:
#   1. Creates .\.venv (uv if installed, else python -m venv) with a
#      Python between 3.11 and 3.13.
#   2. Installs sable (editable) + dev extras; adds the [training] extra
#      unless --no-training is passed.
#   3. Runs `sable setup` — detects the GPU and builds llama.cpp for it
#      (skipped with a warning when git/cmake/a compiler are absent).
#   4. Reports which screen-capture backend is present (Pillow ImageGrab).
#
# Double-click friendly via setup.cmd, or run:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Note: bitsandbytes (part of [training]) is Linux-only per pyproject marker;
# on Windows use --no-training unless you have a supported CUDA toolchain.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Repo = $PSScriptRoot
$Venv = Join-Path $Repo ".venv"

# ---------------------------------------------------------------- arguments
$Training = $true
$BackendArgs = @()
$i = 0
while ($i -lt $args.Count) {
    switch ($args[$i]) {
        "--no-training" { $Training = $false }
        "--backend" {
            $i++
            if ($i -ge $args.Count) { Write-Error "--backend requires a value"; exit 2 }
            $BackendArgs = @("--backend", $args[$i])
        }
        { $_ -in "-h", "--help" } {
            Write-Host "usage: setup.ps1 [--no-training] [--backend cuda|rocm|vulkan|cpu]"
            exit 0
        }
        default { Write-Error "unknown option: $($args[$i]) (see --help)"; exit 2 }
    }
    $i++
}

function Say([string]$m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "warning: $m" -ForegroundColor Yellow }

$HasUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)

# ------------------------------------------------------------- 1. interpreter
Say "1/4 locating a Python (3.11-3.13)"
$Py = $null
if ($HasUv) {
    $Py = (& uv python find ">=3.11,<3.14" 2>$null) | Select-Object -First 1
    if (-not $Py) {
        Say "no suitable interpreter found - installing one with uv"
        & uv python install 3.13 2>$null | Out-Null
        $Py = (& uv python find ">=3.11,<3.14") | Select-Object -First 1
    }
} else {
    foreach ($c in "python3.13", "python3.12", "python3.11", "python") {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        & $cmd.Source -c "import sys; raise SystemExit(not (3,11) <= sys.version_info < (3,14))" 2>$null
        if ($LASTEXITCODE -eq 0) { $Py = $cmd.Source; break }
    }
}
if (-not $Py) {
    Write-Error "error: need Python 3.11-3.13 (or install uv: https://docs.astral.sh/uv/)"
    exit 1
}
Write-Host "interpreter: $Py ($(& $Py --version 2>&1))"

# ------------------------------------------------------------------- 2. venv
Say "2/4 creating venv + installing sable"
$Extras = if ($Training) { "dev,training" } else { "dev" }
if ($HasUv) {
    & uv venv $Venv --python $Py 2>$null | Out-Null
    & uv pip install --python (Join-Path $Venv "Scripts\python.exe") -q "-e" "${Repo}[$Extras]"
    if ($LASTEXITCODE -ne 0) { Warn "uv pip install failed"; exit 1 }
} else {
    $venvPy = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { & $Py -m venv $Venv }
    & $venvPy -m pip install -q --upgrade pip setuptools wheel
    & $venvPy -m pip install -q "-e" "${Repo}[$Extras]"
    if ($LASTEXITCODE -ne 0) { Warn "package install failed (check network/disk)"; exit 1 }
}
$Sable = Join-Path $Venv "Scripts\sable.exe"
if (-not (Test-Path $Sable)) { Write-Error "error: venv install failed - no .venv\Scripts\sable.exe"; exit 1 }
Write-Host "installed  : $(& $Sable --version)"

# --------------------------------------------------------------- 3. llama.cpp
Say "3/4 building llama.cpp for this machine (GPU autodetect)"
$Toolchain = @("git", "cmake") | Where-Object { -not (Get-Command $_ -ErrorAction SilentlyContinue) }
if ($Toolchain) {
    Warn "missing toolchain: $($Toolchain -join ', ') - skipping llama.cpp build."
    Warn "install them (e.g. winget install Kitware.Cake Git) plus a C/C++ compiler and re-run setup.ps1"
} else {
    & $Sable setup @BackendArgs
    if ($LASTEXITCODE -ne 0) {
        Warn "llama.cpp setup failed - serving/training may be limited; re-run setup.ps1 or 'sable setup' after fixing the toolchain"
    }
}

# ------------------------------------------------------- 4. screenshot backend
Say "4/4 checking screen-capture backend for take_screenshot"
& (Join-Path $Venv "Scripts\python.exe") -c "import PIL.ImageGrab" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "capture    : Pillow ImageGrab - OK (Windows native)"
} else {
    Warn "Pillow missing from venv - re-run setup.ps1 (installed via pyproject)"
}

Say "done."
Write-Host "  launch   : .\launch.ps1   (or double-click launch.cmd)"
Write-Host "  stop     : .\launch.ps1 --stop"
Write-Host "  cleanup  : .\cleanup.ps1  (fresh-downloaded state)"
Write-Host "  tests    : .venv\Scripts\pytest"
