# toaster-loop cleanup (Windows) — return the program to a freshly-downloaded state.
#
#   .\cleanup.ps1              # interactive: shows what would be removed, asks
#   .\cleanup.ps1 --yes        # no prompt (also: -y)
#   .\cleanup.ps1 --keep-models
#
# PowerShell port of cleanup.sh (A3). Removes:
#   * TOASTER_LOOP_HOME runtime state (%USERPROFILE%\.toaster-loop by default): traces,
#     skills, mined goldens, buffer, datasets, model registry, logs,
#     serve/launch state, managed llama.cpp build (everything under the home).
#   * Legacy %USERPROFILE%\.unforge state dir from before the rename.
#   * Local dev artifacts in this checkout: .venv\, __pycache__\, egg-info,
#     pytest/mypy/ruff caches, stray runtime dirs/dbs the repo .gitignore
#     lists, and coverage output (.coverage*, htmlcov\).
# Keeps: everything tracked by git (source, docs, tests) — i.e. exactly the
# state right after `git clone`. Never touches foreign installs.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Repo = $PSScriptRoot

# ---------------------------------------------------------------- arguments
$AssumeYes = $false
$KeepModelsArgs = @()
foreach ($a in $args) {
    switch ($a) {
        { $_ -in "--yes", "-y" } { $AssumeYes = $true }
        "--keep-models" { $KeepModelsArgs = @("--keep-models") }
        { $_ -in "-h", "--help" } {
            Write-Host "usage: cleanup.ps1 [--yes|-y] [--keep-models]"
            exit 0
        }
        default { Write-Error "unknown option: $a (see --help)"; exit 2 }
    }
}

# Windows PowerShell 5.1 turns any text a native command writes to stderr into
# a NativeCommandError, which is terminating under $ErrorActionPreference =
# "Stop"; pwsh 7 does not do this. Run native tools inside a child scope pinned
# to Continue so stderr stays plain progress text.
function Native([scriptblock]$Body) {
    & { $ErrorActionPreference = "Continue"; . $Body }
}

$ToasterLoopBin = Join-Path $Repo ".venv\Scripts\toaster-loop.exe"

Write-Host "This returns toaster-loop to a freshly-downloaded state:"
Write-Host "  - wipes the runtime home (traces, datasets, models, logs, llama.cpp)"
Write-Host "  - deletes the local .venv and build artifacts in $Repo"
Write-Host "Source files under git are left untouched."
Write-Host ""

if (-not $AssumeYes) {
    $ans = Read-Host "Continue? [y/N]"
    if ($ans -notin @("y", "Y", "yes", "YES")) { Write-Host "aborted."; exit 0 }
}

# 1. Runtime state (stops toaster-loop-owned server/dashboard first). Uses the venv
#    CLI when present; otherwise a direct directory removal as fallback —
#    guarded to look like an toaster-loop home, same refusal as cleanup.sh.
if (Test-Path $ToasterLoopBin) {
    Write-Host "== wiping runtime home =="
    Native { & $ToasterLoopBin reset --yes @KeepModelsArgs }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "warning: 'toaster-loop reset' failed; continuing with repo cleanup"
    }
} else {
    Write-Host "== runtime home: venv already gone - removing $HOME\.toaster-loop directly =="
    $HomeDir = if ($env:TOASTER_LOOP_HOME) { $env:TOASTER_LOOP_HOME } else { Join-Path $HOME ".toaster-loop" }
    # Portable (Windows PowerShell 5.1 has no ternary operator).
    $Resolved = Resolve-Path -ErrorAction SilentlyContinue $HomeDir
    if ($Resolved) { $HomeNorm = $Resolved.Path } else { $HomeNorm = $HomeDir }
    $LooksLikeHome = ($HomeNorm -like (Join-Path $HOME ".toaster-loop")) -or
                     ($HomeNorm -like (Join-Path $HOME ".toaster-loop-*"))
    if ($LooksLikeHome) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $HomeDir
    } else {
        Write-Warning "refusing to delete unexpected TOASTER_LOOP_HOME=$HomeDir"
    }
}

# 1b. Legacy pre-rename state dir (%USERPROFILE%\.unforge), if any survived.
if (-not $env:UNFORGE_HOME) {
    $Legacy = Join-Path $HOME ".unforge"
    if (Test-Path $Legacy) {
        Write-Host "== removing legacy $Legacy state =="
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $Legacy
    }
}

# 2. Repo-local artifacts (the ones .gitignore declares as generated).
Write-Host "== cleaning repo checkout =="
foreach ($d in ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "toaster_loop.egg-info",
               "unforge.egg-info", "None",
               "adapter", "outputs", "train_runs", "datasets", "models", "logs", "htmlcov") {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Repo $d)
}
Get-ChildItem -Path $Repo -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
foreach ($f in "serve_state.json", "serve_state.json.tmp", "config.yaml", ".env") {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Repo $f)
}
# Stray databases and coverage artifacts anywhere in the tree.
Get-ChildItem -Path $Repo -Recurse -File -Include "*.db", "*.db-wal", "*.db-shm", ".coverage*" `
    -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "toaster-loop is back to a freshly-downloaded state."
Write-Host "To start using it again: .\setup.ps1"
