# toaster-loop one-click setup (Windows) — safe to double-click or run from a terminal:
#
#   .\setup.ps1                          # full setup (venv + package + training
#                                        #  extras + llama.cpp build for this GPU)
#   .\setup.ps1 --no-training            # skip the heavy torch training stack
#   .\setup.ps1 --backend cuda|rocm|vulkan|cpu   # override GPU autodetect
#
# PowerShell port of setup.sh (A3). What it does:
#   1. Creates .\.venv (uv if installed, else python -m venv) with a
#      Python between 3.11 and 3.13.
#   2. Installs toaster-loop (editable) + dev extras; adds the [training] extra
#      unless --no-training is passed.
#   3. Runs `toaster-loop setup` — detects the GPU and builds llama.cpp for it
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

# Windows PowerShell 5.1 turns any text a native command writes to stderr into
# a NativeCommandError, which is terminating under $ErrorActionPreference =
# "Stop" (even when redirected with 2>$null); pwsh 7 does not do this. Run every
# native tool inside a child scope pinned to Continue so stderr stays plain
# progress text, and check $LASTEXITCODE for real failures.
function Native([scriptblock]$Body) {
    & { $ErrorActionPreference = "Continue"; . $Body }
}

$HasUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)

# ------------------------------------------------------------- 1. interpreter
Say "1/4 locating a Python (3.11-3.13)"
$Py = $null
if ($HasUv) {
    $Py = (Native { & uv python find ">=3.11,<3.14" }) | Select-Object -First 1
    if (-not $Py) {
        Say "no suitable interpreter found - installing one with uv"
        Native { & uv python install 3.13 } | Out-Null
        $Py = (Native { & uv python find ">=3.11,<3.14" }) | Select-Object -First 1
    }
} else {
    foreach ($c in "python3.13", "python3.12", "python3.11", "python") {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        Native { & $cmd.Source -c "import sys; raise SystemExit(not (3,11) <= sys.version_info < (3,14))" } 2>$null
        if ($LASTEXITCODE -eq 0) { $Py = $cmd.Source; break }
    }
}
if (-not $Py) {
    Write-Error "error: need Python 3.11-3.13 (or install uv: https://docs.astral.sh/uv/)"
    exit 1
}
$PyVersion = (Native { & $Py --version }) -join " "
Write-Host "interpreter: $Py ($PyVersion)"

# ------------------------------------------------------------------- 2. venv
Say "2/4 creating venv + installing toaster-loop"
$Extras = if ($Training) { "dev,training" } else { "dev" }
if ($HasUv) {
    Native { & uv venv $Venv --python $Py } 2>$null | Out-Null
} else {
    $venvPy = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $venvPy)) { Native { & $Py -m venv $Venv } }
    Native { & $venvPy -m pip install -q --upgrade pip setuptools wheel }
}

# torch build flavor must match THIS machine's accelerator (plain PyPI ships
# CUDA on Windows too; Intel Arc needs the xpu index). See
# toaster-loop/core/torchenv.py. Install torch FIRST from the right index so the
# [training] extras resolve against it.
if ($Training) {
    $env:PYTHONPATH = $Repo
    $venvPy = Join-Path $Venv "Scripts\python.exe"
    $Flavor = ((Native { & $venvPy -c "from toaster_loop.core.torchenv import detect_flavor; print(detect_flavor().flavor)" } ) | Select-Object -Last 1)
    if (-not $Flavor) { $Flavor = "default" }
    $FlavorIdx = ((Native { & $venvPy -c "from toaster_loop.core.torchenv import detect_flavor; print(detect_flavor().index_url or '')" } ) | Select-Object -Last 1)
    Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
    Write-Host "torch      : $Flavor build for this hardware"
    if ($HasUv) {
        if ($Flavor -eq "default") {
            Native { & uv pip install --python $venvPy -q torch }
        } else {
            Native { & uv pip install --python $venvPy -q torch --torch-backend $Flavor --extra-index-url https://download.pytorch.org/whl }
        }
    } elseif ($FlavorIdx) {
        Native { & $venvPy -m pip install -q torch --index-url $FlavorIdx }
    } else {
        Native { & $venvPy -m pip install -q torch }
    }
    if ($LASTEXITCODE -ne 0) { Warn "torch install failed ($Flavor)"; exit 1 }
}

if ($HasUv) {
    Native { & uv pip install --python (Join-Path $Venv "Scripts\python.exe") -q "-e" "${Repo}[$Extras]" }
    if ($LASTEXITCODE -ne 0) { Warn "uv pip install failed"; exit 1 }
} else {
    Native { & $venvPy -m pip install -q "-e" "${Repo}[$Extras]" }
    if ($LASTEXITCODE -ne 0) { Warn "package install failed (check network/disk)"; exit 1 }
}
$ToasterLoopBin = Join-Path $Venv "Scripts\toaster-loop.exe"
if (-not (Test-Path $ToasterLoopBin)) { Write-Error "error: venv install failed - no .venv\Scripts\toaster-loop.exe"; exit 1 }
$ToasterLoopVersion = (Native { & $ToasterLoopBin --version }) -join " "
Write-Host "installed  : $ToasterLoopVersion"

# --------------------------------------------------------------- 3. llama.cpp
Say "3/4 building llama.cpp for this machine (GPU autodetect)"

# Offer to install missing build prerequisites via winget. winget handles its
# own elevation prompt; PATH is refreshed afterwards so the new tools are
# visible without opening a fresh terminal.
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Ensure-Tool([string]$name, [string]$wingetId) {
    if (Get-Command $name -ErrorAction SilentlyContinue) { return $true }
    Warn "$name not found."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Warn "winget unavailable - install $name manually ($wingetId) and re-run setup.ps1"
        return $false
    }
    $ans = Read-Host "Install $name via winget ($wingetId)? [Y/n]"
    if ($ans -in @("", "y", "Y", "yes", "YES")) {
        Native { & winget install --id $wingetId -e --accept-source-agreements --accept-package-agreements }
        Refresh-Path
        if (Get-Command $name -ErrorAction SilentlyContinue) { return $true }
        Warn "$name installed but not on PATH yet - close and reopen this terminal, then re-run setup.ps1"
    }
    return $false
}

# NVIDIA CUDA toolkit: needed only when an NVIDIA GPU is present and nvcc is
# absent. Note VS 2026 + CUDA requires the *newest* toolkit (12.8 cannot build
# against a VS 2026 host compiler); toaster-loop's build pins the newest installed one.
$HasNvGpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$HasNvcc = [bool](Get-Command nvcc -ErrorAction SilentlyContinue) -or
          [bool]$env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))
if ($HasNvGpu -and -not $HasNvcc) {
    Warn "NVIDIA GPU detected but no CUDA toolkit (nvcc) found."
    $null = Ensure-Tool nvcc "Nvidia.CUDA"
    # winget puts nvcc under the toolkit dir, not PATH; accept either.
    if (-not ((Get-Command nvcc -ErrorAction SilentlyContinue) -or
              (Test-Path "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe"))) {
        Warn "continuing without CUDA - toaster-loop setup will fall back to another backend"
    }
}

$ToolchainOk = $true
foreach ($t in @("git", "Git.Git"), @("cmake", "Kitware.CMake")) {
    if (-not (Ensure-Tool $t[0] $t[1])) { $ToolchainOk = $false }
}
if (-not $ToolchainOk) {
    Warn "toolchain incomplete - skipping llama.cpp build; re-run setup.ps1 after installing git/cmake"
} else {
    Native { & $ToasterLoopBin setup @BackendArgs }
    if ($LASTEXITCODE -ne 0) {
        Warn "llama.cpp setup failed - serving/training may be limited; re-run setup.ps1 or 'toaster-loop setup' after fixing the toolchain"
    }
}

# ------------------------------------------------------- 4. screenshot backend
Say "4/4 checking screen-capture backend for take_screenshot"
Native { & (Join-Path $Venv "Scripts\python.exe") -c "import PIL.ImageGrab" } 2>$null
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
