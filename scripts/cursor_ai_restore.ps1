param(
    [string]$Id = "",
    [switch]$List,
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

$checkpointPath = Join-Path $repoRoot ".cursor\ai_checkpoints.json"

# Also check old single-checkpoint file for migration
$oldPath = Join-Path $repoRoot ".cursor\ai_checkpoint.json"
if (-not (Test-Path $checkpointPath) -and (Test-Path $oldPath)) {
    # Migrate old format
    $oldData = Get-Content -Raw -Path $oldPath | ConvertFrom-Json
    $migrated = @($oldData)
    $json = ($migrated | ConvertTo-Json -Depth 4)
    if ($migrated.Count -eq 1) { $json = "[$json]" }
    $json | Set-Content -Encoding UTF8 -Path $checkpointPath
    Remove-Item $oldPath -Force
    Write-Host "Migrated old checkpoint to new stack format."
}

if (-not (Test-Path $checkpointPath)) {
    throw "No checkpoints found. Run scripts\cursor_ai_checkpoint.ps1 first."
}

$raw = Get-Content -Raw -Path $checkpointPath
$stack = @($raw | ConvertFrom-Json)

if ($stack.Count -eq 0) {
    throw "Checkpoint stack is empty."
}

# --- List mode ---
if ($List) {
    Write-Host ""
    Write-Host "=== Checkpoint Stack ($($stack.Count) total) ==="
    Write-Host ""
    for ($i = $stack.Count - 1; $i -ge 0; $i--) {
        $cp = $stack[$i]
        $idx = $i + 1
        $hasStash = if ($cp.stashRef -and ([string]$cp.stashRef).Trim().Length -gt 0) { "+" } else { "-" }
        Write-Host ("  #{0}  [{1}]  {2}  {3}  (stash: {4})" -f $idx, $cp.timestamp, $cp.message, $cp.head.Substring(0,7), $hasStash)
    }
    Write-Host ""
    Write-Host "Use: .\scripts\cursor_ai_restore.ps1 -Id <timestamp>"
    Write-Host "  or: .\scripts\cursor_ai_restore.ps1            (restores most recent)"
    return
}

# --- Pick checkpoint ---
$target = $null
$targetIdx = -1

if ($Id -and $Id.Trim().Length -gt 0) {
    # Find by id/timestamp
    for ($i = 0; $i -lt $stack.Count; $i++) {
        if ([string]$stack[$i].id -eq $Id -or [string]$stack[$i].timestamp -eq $Id) {
            $target = $stack[$i]
            $targetIdx = $i
            break
        }
    }
    if (-not $target) {
        throw "Checkpoint with id '$Id' not found. Use -List to see available checkpoints."
    }
} else {
    # Default: most recent (last in array)
    $targetIdx = $stack.Count - 1
    $target = $stack[$targetIdx]
}

$head = [string]$target.head
$stashRef = [string]$target.stashRef
$msg = [string]$target.message
$ts = [string]$target.timestamp

Write-Host ""
Write-Host "Restoring to checkpoint: '${msg}' (${ts})"
Write-Host "  HEAD -> ${head}"

# Find the stash index by matching the stash message
$stashIndex = $null
if ($stashRef -and $stashRef.Trim().Length -gt 0) {
    $stashMsg = [string]$target.stashMsg
    $stashListLines = @(git stash list)
    for ($i = 0; $i -lt $stashListLines.Count; $i++) {
        if ($stashListLines[$i] -match $stashMsg) {
            $stashIndex = $i
            break
        }
    }
    if ($null -eq $stashIndex) {
        Write-Warning "Could not find matching stash by message. Will try by ref."
    }
}

# Reset working tree
git reset --hard $head | Out-Host
git clean -fd | Out-Host

# Re-apply stash if present
if ($stashRef -and $stashRef.Trim().Length -gt 0) {
    if ($null -ne $stashIndex) {
        $stashName = "stash@{${stashIndex}}"
        Write-Host "Re-applying stash ${stashName}"
        git stash apply $stashName | Out-Host
        if (-not $KeepStash) {
            try { git stash drop $stashName | Out-Host } catch {
                Write-Warning "Could not drop stash. Run: git stash list / git stash drop"
            }
        }
    } else {
        Write-Host "Re-applying stash by ref ${stashRef}"
        git stash apply $stashRef | Out-Host
        if (-not $KeepStash) {
            try { git stash drop $stashRef | Out-Host } catch {
                Write-Warning "Could not drop stash. Run: git stash list / git stash drop"
            }
        }
    }
} else {
    Write-Host "No stash for this checkpoint."
}

# Remove this checkpoint and all newer ones from the stack
$newStack = @()
for ($i = 0; $i -lt $targetIdx; $i++) {
    $newStack += $stack[$i]
}

if ($newStack.Count -gt 0) {
    $json = ($newStack | ConvertTo-Json -Depth 4)
    if ($newStack.Count -eq 1) { $json = "[$json]" }
    $json | Set-Content -Encoding UTF8 -Path $checkpointPath
} else {
    Remove-Item $checkpointPath -Force
}

$remaining = $newStack.Count
Write-Host ""
Write-Host "Restore complete. Remaining checkpoints: ${remaining}"
