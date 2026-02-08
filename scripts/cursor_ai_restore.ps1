param(
    [string]$Id = "",
    [switch]$List,
    [switch]$KeepStash
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

$checkpointPath = Join-Path $repoRoot ".cursor\ai_checkpoints.json"

# Migrate old single-checkpoint file if needed
$oldPath = Join-Path $repoRoot ".cursor\ai_checkpoint.json"
if (-not (Test-Path $checkpointPath) -and (Test-Path $oldPath)) {
    $oldData = ConvertFrom-Json -InputObject (Get-Content -Raw -Path $oldPath)
    $arr = @($oldData)
    $json = ConvertTo-Json -InputObject $arr -Depth 4
    Set-Content -Encoding UTF8 -Path $checkpointPath -Value $json
    Remove-Item $oldPath -Force
    Write-Host "Migrated old checkpoint to new stack format."
}

if (-not (Test-Path $checkpointPath)) {
    throw "No checkpoints found. Run scripts\cursor_ai_checkpoint.ps1 first."
}

# --- Load stack (PS 5.1 safe) ---
$raw = Get-Content -Raw -Path $checkpointPath
$parsed = ConvertFrom-Json -InputObject $raw
$stack = New-Object System.Collections.ArrayList
if ($parsed -is [System.Array]) {
    foreach ($item in $parsed) { $stack.Add($item) | Out-Null }
} else {
    $stack.Add($parsed) | Out-Null
}

if ($stack.Count -eq 0) { throw "Checkpoint stack is empty." }

# --- List mode ---
if ($List) {
    Write-Host ""
    Write-Host "=== Checkpoint Stack ($($stack.Count) total) ==="
    Write-Host ""
    for ($i = $stack.Count - 1; $i -ge 0; $i--) {
        $cp = $stack[$i]
        $idx = $i + 1
        $sr = [string]$cp.stashRef
        $hasStash = if ($sr -and $sr.Trim().Length -gt 0) { "yes" } else { "no" }
        $shortHead = ([string]$cp.head).Substring(0, 7)
        Write-Host ("  #{0}  [{1}]  {2}  {3}  (stash: {4})" -f $idx, $cp.id, $cp.message, $shortHead, $hasStash)
    }
    Write-Host ""
    Write-Host "Restore most recent:  .\scripts\cursor_ai_restore.ps1"
    Write-Host "Restore by id:        .\scripts\cursor_ai_restore.ps1 -Id <id>"
    return
}

# --- Pick checkpoint ---
$target = $null
$targetIdx = -1

if ($Id -and $Id.Trim().Length -gt 0) {
    for ($i = 0; $i -lt $stack.Count; $i++) {
        if ([string]$stack[$i].id -eq $Id) {
            $target = $stack[$i]
            $targetIdx = $i
            break
        }
    }
    if (-not $target) {
        throw "Checkpoint '$Id' not found. Use -List to see available."
    }
} else {
    $targetIdx = $stack.Count - 1
    $target = $stack[$targetIdx]
}

$head = [string]$target.head
$stashRef = [string]$target.stashRef
$msg = [string]$target.message
$ts = [string]$target.id

Write-Host ""
Write-Host "Restoring to checkpoint: '${msg}' (${ts})"
Write-Host "  HEAD -> ${head}"

# --- Find stash by message in stash list ---
$stashIndex = $null
if ($stashRef -and $stashRef.Trim().Length -gt 0) {
    $stashMsg = [string]$target.stashMsg
    $stashListLines = @(git stash list)
    for ($i = 0; $i -lt $stashListLines.Count; $i++) {
        if ($stashListLines[$i] -match [regex]::Escape($stashMsg)) {
            $stashIndex = $i
            break
        }
    }
    if ($null -eq $stashIndex) {
        Write-Warning "Could not find matching stash by message. Will try by ref."
    }
}

# --- Reset working tree ---
git reset --hard $head | Out-Host
git clean -fd | Out-Host

# --- Re-apply stash ---
if ($stashRef -and $stashRef.Trim().Length -gt 0) {
    if ($null -ne $stashIndex) {
        $stashName = "stash@{${stashIndex}}"
        Write-Host "Re-applying stash ${stashName}"
        git stash apply $stashName | Out-Host
        if (-not $KeepStash) {
            try { git stash drop $stashName | Out-Host } catch {
                Write-Warning "Could not drop stash. Run: git stash list"
            }
        }
    } else {
        Write-Host "Re-applying stash by ref ${stashRef}"
        git stash apply $stashRef | Out-Host
        if (-not $KeepStash) {
            try { git stash drop $stashRef | Out-Host } catch {
                Write-Warning "Could not drop stash. Run: git stash list"
            }
        }
    }
} else {
    Write-Host "No stash for this checkpoint."
}

# --- Trim stack: remove this checkpoint and all newer ones ---
$newStack = New-Object System.Collections.ArrayList
for ($i = 0; $i -lt $targetIdx; $i++) {
    $newStack.Add($stack[$i]) | Out-Null
}

if ($newStack.Count -gt 0) {
    $arr = $newStack.ToArray()
    $json = ConvertTo-Json -InputObject $arr -Depth 4
    Set-Content -Encoding UTF8 -Path $checkpointPath -Value $json
} else {
    Remove-Item -ErrorAction SilentlyContinue $checkpointPath -Force
}

$remaining = $newStack.Count
Write-Host ""
Write-Host "Restore complete. Remaining checkpoints: ${remaining}"
