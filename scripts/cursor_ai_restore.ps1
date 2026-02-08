param(
    [switch]$KeepStash
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Required command not found: $Name"
    }
}

Require-Command git

$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "Not inside a git repository."
}

Set-Location $repoRoot

$checkpointPath = Join-Path $repoRoot ".cursor\\ai_checkpoint.json"
if (-not (Test-Path $checkpointPath)) {
    throw "Checkpoint file not found: $checkpointPath. Run scripts\\cursor_ai_checkpoint.ps1 first."
}

$data = Get-Content -Raw -Path $checkpointPath | ConvertFrom-Json
$head = [string]$data.head
$stashRef = [string]$data.stashRef
$stashDropRef = [string]$data.stashDropRef

if (-not $head) {
    throw "Checkpoint is missing 'head'."
}

Write-Host "Restoring tracked files to checkpoint HEAD $head"
git reset --hard $head | Out-Host
git clean -fd | Out-Host

if ($stashRef -and $stashRef.Trim().Length -gt 0) {
    Write-Host "Re-applying stashed pre-edit changes ($stashRef)"
    git stash apply $stashRef | Out-Host

    if (-not $KeepStash) {
        try {
            if ($stashDropRef -and $stashDropRef.Trim().Length -gt 0) {
                git stash drop $stashDropRef | Out-Host
            } else {
                git stash drop $stashRef | Out-Host
            }
        } catch {
            Write-Warning ("Could not drop stash automatically. You can remove it manually via: git stash list / git stash drop")
        }
    } else {
        Write-Host "KeepStash enabled; not dropping stash."
    }
} else {
    Write-Host "No stash recorded in checkpoint; restore complete."
}

Write-Host "Restore complete."
