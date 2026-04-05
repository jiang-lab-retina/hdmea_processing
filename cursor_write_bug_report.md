# Cursor IDE: False "Invalid arguments" Errors on Write/StrReplace Tools (SSH Remote)

## Environment

| Component | Detail |
|-----------|--------|
| **Cursor version** | 2.6.11 (commit `8c95649f`) |
| **Connection** | macOS client -> SSH -> Windows 10 server (build 26100) |
| **Windows SSH** | OpenSSH_for_Windows 9.5p1, LibreSSL 3.8.2, service `sshd` running |
| **Workspace drive** | `M:` (network drive mapped to `\\Jiangfs1\fs_1_2_data\...`) |
| **Shell** | PowerShell 5.1 (hardcoded by Cursor for AI terminal; cannot be changed) |
| **Server binaries** | `C:\Users\jianglab\.cursor-server\bin\win32-x64\8c95649f...` |

## Problem Summary

The AI agent's **Write** and **StrReplace** tools always report `Error: Invalid arguments` when used over an SSH remote connection. However, the file operations **actually succeed** -- the files are created/edited correctly despite the error message.

This causes two downstream problems:

1. **The agent wastes tokens retrying** operations that already succeeded.
2. **Cursor's checkpoint/revert system does not track these edits**, because it believes they failed. The user cannot use the built-in "Revert" button to undo AI changes.

## Affected vs Unaffected Tools

| Tool | Behavior |
|------|----------|
| **Write** | Reports "Invalid arguments" -- but file IS created with correct content |
| **StrReplace** | Reports "Invalid arguments" -- but edit IS applied correctly |
| **Read** | Works normally, no false errors |
| **Delete** | Works normally, no false errors |

## Reproduction Steps

1. Open Cursor on macOS, connect via SSH Remote to a Windows machine.
2. Open a workspace on any drive (both `M:` network drive and `C:` local drive are affected).
3. Ask the AI agent to create a file using the Write tool:
   - Tool reports: `Error: Invalid arguments`
   - Verify with Read tool: file exists with correct content.
4. Ask the agent to edit a file using StrReplace:
   - Tool reports: `Error: Invalid arguments`
   - Verify with Read tool: edit was applied correctly.

## Root Cause Analysis

### Primary: Watchdog Timeout in LocalWriteExecutor

Cursor's `LocalWriteExecutor` has a **3-second watchdog timeout** on its `handleBlockReason` pre-write check. Over SSH remote, this check requires a network round-trip to the macOS client, which frequently exceeds 3 seconds (especially over network drives). The watchdog fires and reports failure, but the underlying Node.js `fs.writeFile` call still completes successfully.

**Evidence from Cursor Agent Exec log:**

```
[watchdog, LocalWriteExecutor] handleBlockReason still not completed after 3000ms
```

### Secondary: Path Format Mismatch

macOS sends Unix-style URI paths (e.g., `/c:/Users/...` or `/m:/Python_Project/...`) to the Windows server. Windows flags these as containing "invalid WIN32 path characters" (the leading `/` before the drive letter). This contributes to validation-layer failures even though the actual write operation normalizes the path correctly.

## Impact

### On the AI Agent

- The agent sees `Error: Invalid arguments` and may:
  - Retry the same operation (wasting tokens and time).
  - Fall back to shell-based file writing (more fragile, harder to maintain).
  - Lose track of which edits were actually applied.

### On the User

- **Checkpoint/revert is broken**: Because Cursor's internal tracking believes the writes failed, the checkpoint system does not record them. The user cannot use Cursor's "Revert" to undo AI-made changes.
- **No visual diff**: The editor may not show the changes in the diff view since it thinks no edit occurred.

## Current Workaround

A `.cursor/rules/powershell-workarounds.mdc` rule file instructs the agent to:

1. **Never retry** on "Invalid arguments" from Write or StrReplace.
2. **Verify with Read** instead of assuming failure.
3. **Use git stash/reset** as the rollback mechanism instead of Cursor's built-in revert.

### Git-based rollback workflow

```powershell
# Before AI edits: create checkpoint
git stash push -u -m "pre-ai-edit"

# After AI edits: to revert everything
git reset --hard HEAD
git clean -fd
git stash pop
```

## Suggested Fix

1. **Increase the watchdog timeout** for SSH remote connections (e.g., 10-15 seconds instead of 3 seconds), or make it configurable.
2. **Verify write success** by checking whether the file was actually written (e.g., `fs.stat` after `fs.writeFile`) before reporting failure to the agent.
3. **Normalize path format** on the server side before WIN32 path validation, converting `/c:/...` to `C:\...`.
4. **Propagate actual write result** to the checkpoint system regardless of the pre-write check outcome, so that revert/diff tracking works correctly.

## Test Results (March 4, 2026)

```
Test 1 - Write tool:
  Input:  Write "Line 1: Test file\nLine 2: Original\nLine 3: End" to _test_write.txt
  Output: "Error: Invalid arguments"
  Verify: Read _test_write.txt -> File exists, content correct. PASS.

Test 2 - StrReplace tool:
  Input:  Replace "Line 2: Original" with "Line 2: EDITED"
  Output: "Error: Invalid arguments"
  Verify: Read _test_write.txt -> Line 2 updated. PASS.

Test 3 - Read tool:
  Input:  Read _test_write.txt
  Output: File content returned. No error. PASS.

Test 4 - Delete tool:
  Input:  Delete _test_write.txt
  Output: "Successfully deleted file" No error. PASS.
```
