param(
    [string]$Message = "pre-ai-edit"
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

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stashMessage = "cursor-ai-checkpoint:$timestamp:$Message"

$status = git status --porcelain
$hasChanges = ($status -and $status.Count -gt 0)

$checkpointDir = Join-Path $repoRoot ".cursor"
if (-not (Test-Path $checkpointDir)) {
    New-Item -ItemType Directory -Path $checkpointDir | Out-Null
}

$checkpointPath = Join-Path $checkpointDir "ai_checkpoint.json"

$head = (git rev-parse HEAD).Trim()
$stashRef = ""
$stashDropRef = ""

if ($hasChanges) {
    # Include untracked files too.
    git stash push -u -m $stashMessage | Out-Host
    $stashRef = (git rev-parse --verify refs/stash).Trim()
    $stashDropRef = "stash@{0}"
} else {
    Write-Host "Working tree clean; no stash created. Recording HEAD only."
}

$data = @{
    repoRoot  = $repoRoot
    head      = $head
    stashRef  = $stashRef
    stashDropRef = $stashDropRef
    message   = $stashMessage
    timestamp = $timestamp
}

$json = ($data | ConvertTo-Json -Depth 4)
$json | Set-Content -Encoding UTF8 -Path $checkpointPath

Write-Host "Checkpoint saved to $checkpointPath"
