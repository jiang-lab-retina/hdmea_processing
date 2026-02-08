param(
    [string]$Message = "pre-ai-edit"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Required command not found: $Name" }
}

Require-Command git

$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) { throw "Not inside a git repository." }
Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stashMessage = "cursor-ai-checkpoint:${timestamp}:${Message}"

$statusLines = @(git status --porcelain)
$hasChanges = ($statusLines.Count -gt 0)

$checkpointDir = Join-Path $repoRoot ".cursor"
if (-not (Test-Path $checkpointDir)) {
    New-Item -ItemType Directory -Path $checkpointDir | Out-Null
}
$checkpointPath = Join-Path $checkpointDir "ai_checkpoints.json"

# --- Load existing stack (PS 5.1 safe) ---
$stack = New-Object System.Collections.ArrayList
if (Test-Path $checkpointPath) {
    $raw = Get-Content -Raw -Path $checkpointPath
    if ($raw -and $raw.Trim().Length -gt 0) {
        $parsed = ConvertFrom-Json -InputObject $raw
        if ($parsed -is [System.Array]) {
            foreach ($item in $parsed) { $stack.Add($item) | Out-Null }
        } else {
            $stack.Add($parsed) | Out-Null
        }
    }
}

# --- Create stash if needed ---
$head = (git rev-parse HEAD).Trim()
$stashRef = ""

if ($hasChanges) {
    git stash push -u -m $stashMessage | Out-Host
    $stashRef = (git rev-parse --verify refs/stash).Trim()
} else {
    Write-Host "Working tree clean; no stash created. Recording HEAD only."
}

# --- Add entry ---
$entry = [ordered]@{
    id       = $timestamp
    message  = $Message
    head     = $head
    stashRef = $stashRef
    stashMsg = $stashMessage
}
$stack.Add($entry) | Out-Null

# --- Save stack (PS 5.1 safe: use -InputObject to serialize as array) ---
$arr = $stack.ToArray()
$json = ConvertTo-Json -InputObject $arr -Depth 4
Set-Content -Encoding UTF8 -Path $checkpointPath -Value $json

$count = $stack.Count
Write-Host ""
Write-Host "Checkpoint #${count} saved: '${Message}' (${timestamp})"
Write-Host "Total checkpoints: ${count}"
Write-Host ""
Write-Host "To restore, run: Terminal > Run Task > Cursor AI: Restore"
