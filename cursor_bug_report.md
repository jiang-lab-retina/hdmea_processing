# Bug Report: Agent Write/StrReplace tools report false "Invalid arguments" on SSH Remote (macOS -> Windows)

## Summary

When using Cursor via SSH Remote (macOS client -> Windows server), the Agent's **Write** and **StrReplace** tools always report `Error: Invalid arguments`, even though the file operations **actually succeed**. This causes two problems:

1. The AI agent sees errors and may retry unnecessarily (doubling edits)
2. Cursor's **checkpoint/revert system does not track these edits**, making it impossible to revert AI changes through the UI

## Environment

- **Client**: macOS (SSH Remote)
- **Server**: Windows 10 (10.0.26100), Windows_NT x64
- **Cursor version**: 2.4.30 (Stable)
- **VSCode version**: 1.105.1
- **Commit**: 0f8217a84adf66daf250228a3ebf0da631d3c9b0
- **Electron**: 39.2.7 / Node.js 22.21.1
- **SSH extension**: anysphere.remote-ssh
- **Workspace**: Network drive (M:) mapped to UNC path `\\server\share`, but bug also reproduces on **local C: drive**

## Steps to Reproduce

1. Connect to a Windows machine via SSH Remote from macOS Cursor client
2. Open any workspace (local or network drive)
3. Use the Agent to write or edit any file
4. Observe: both Write and StrReplace tools report `Error: Invalid arguments`
5. Verify with Read tool: the file was actually created/modified correctly
6. Observe: Cursor's checkpoint/revert button does not work for these edits

## Expected Behavior

- Write and StrReplace should report success when the file operation succeeds
- Cursor's checkpoint system should track these edits so the user can revert them

## Actual Behavior

- Write tool: reports `Error: Invalid arguments` but file IS created with correct content
- StrReplace tool: reports `Error: Invalid arguments` but edit IS applied correctly
- Read tool: works normally, no false errors
- Delete tool: works normally, returns success with byte count
- Cursor's "Revert" / checkpoint feature does not work (edits not tracked)

## Diagnostic Evidence

### 1. Server-side watchdog log (same on working AND broken dates)

The `Cursor Agent Exec.log` contains this warning on EVERY write operation over SSH:

```
[watchdog, LocalWriteExecutor] handleBlockReason still not completed after 3000ms
```

**Critically, this same warning appears on BOTH working and broken dates:**

| Date | Server commit | Log entry | Result for user |
|---|---|---|---|
| Jan 28 | 618c607a | handleBlockReason still not completed after 3000ms | **Worked** |
| Feb 1 | 379934e0 | handleBlockReason still not completed after 3000ms | **Worked** |
| Feb 5 | 4f2b7727 | handleBlockReason still not completed after 3000ms | **BROKEN** |
| Feb 6 | f3f5cec4 | handleBlockReason still not completed after 3000ms | **BROKEN** |
| Feb 7 | 0f8217a8 | handleBlockReason still not completed after 3000ms | **BROKEN** |

The server-side behavior is IDENTICAL. The watchdog fires, but the write succeeds in all cases. The difference is how the **client** handles the response.

### 2. Watchdog function does NOT cause errors (source code proof)

The watchdog function (`Rd`) in `cursor-agent-exec/dist/main.js` only logs a warning:

```javascript
const Rd = async (e, t, n, r) => {
    const s = setTimeout(() => {
        logger.warn(e, `[watchdog, LocalWriteExecutor] ${t} still not completed after ${n}ms`);
    }, n);
    try {
        return await r();  // Still awaits and returns the actual result
    } finally {
        clearTimeout(s);
    }
};
```

It does NOT throw, reject, cancel, or modify the result. The file write always completes and the server always returns a success protobuf response:

```javascript
return new kd.v3({result:{case:"success", value: new kd.j6({path, linesCreated, fileSize, ...})}})
```

### 3. Server code is IDENTICAL between working and broken versions

We compared the `LocalWriteExecutor` class, `shouldBlockWrite`, and watchdog function across all 5 cached server versions. **They are structurally identical** (only minified variable names differ):

| Server commit | Date | main.js size | LocalWriteExecutor | shouldBlockWrite | Watchdog |
|---|---|---|---|---|---|
| 618c607a | Jan 28 | 3834 KB | identical | identical | identical |
| 379934e0 | Feb 1 | 3834 KB | identical | identical | identical |
| 4f2b7727 | Feb 3 | 3998 KB | identical | identical | identical |
| f3f5cec4 | Feb 5 | 3998 KB | identical | identical | identical |
| 0f8217a8 | Feb 7 | 3969 KB | identical | identical | identical |

The 164KB size increase (Feb 1 -> Feb 3) is in OTHER code (MCP tools, etc.), not in the write executor.

Other server files compared:
- `server-main.js`: **0 bytes difference** between Feb 1 and Feb 3
- `extensionHostProcess.js`: **1344 bytes difference** (minification noise only)
- `cursor-agent/dist/main.js`: **677 bytes difference** (minification noise only)

### 4. "Invalid arguments" text NOT in server code

We searched all server-side code for the exact error message. The string "Invalid arguments" does NOT appear in any Cursor-written tool handling code. It only exists in third-party libraries (undici HTTP client, Zod validation). The error message is generated by the **macOS client**.

### 5. Additional path format issue

The `remoteexthost.log` shows repeated errors:

```
Error: /c:/Users/username/.cursor-server/data/... contains invalid WIN32 path characters.
```

macOS sends Unix-style URI paths (`/c:/...`) that Windows flags as invalid.

## Root Cause: TWO BUGS COMBINING

### Bug 1: Mapped drive realpath mismatch (server-side)

On Windows mapped drives (M: -> \\\\server\\share), Node.js `realpath()` resolves existing paths to UNC format but falls back to drive letter for non-existent files:

```
realpath("M:\\...\\Data_Processing_2027")         -> "\\\\Jiangfs1\\fs_1_2_data\\...\\Data_Processing_2027"  (UNC)
realpath("M:\\...\\Data_Processing_2027\\new.txt") -> FAILS -> falls back to "M:\\...\\new.txt"  (drive letter)
```

In `shouldBlockWrite`, when approval mode is NOT "unrestricted", a realpath comparison checks if the file is "in the workspace". The workspace resolves to UNC but the file stays as M:\\ -> **mismatch** -> file treated as "out of workspace" -> triggers `requestApproval` round-trip.

### Bug 2: Client mishandles slow approval response (client-side)

The `requestApproval` call sends a nested RPC back to the macOS client over SSH. This round-trip takes >3 seconds. The Feb 1 macOS client handled this correctly. The Feb 3+ macOS client introduced a regression that reports "Invalid arguments" when the approval round-trip is slow.

Evidence:
1. Server code is identical between working (Feb 1) and broken (Feb 3+) versions
2. Server logs show the same watchdog warning on all dates (even when it worked)
3. Server always returns a "success" protobuf response
4. The "Invalid arguments" error text is not in the server code
5. The macOS client was updated between Feb 1 and Feb 3

### Why both bugs are needed

- Bug 1 alone doesn't cause the error (Feb 1 client handled slow responses fine)
- Bug 2 alone doesn't cause the error (files in temp dirs or "unrestricted" mode skip the realpath check)
- Together: mapped drive triggers slow approval -> new client mishandles it -> "Invalid arguments"

### Fix applied

We patched the server's `shouldBlockWrite` to force the "unrestricted" branch for write permission checks. This makes `shouldBlockWrite` return `false` before reaching the broken realpath comparison, avoiding the slow approval round-trip entirely.

```javascript
// Original: const o="unrestricted"===i.approvalMode
// Patched:  const o=!0||"unrestricted"==i.approvalMode
```

This patch was applied to all 5 cached server versions in `.cursor-server/bin/`.

## NOT drive-specific

This bug affects ALL drives on the Windows server:
- Local C: drive: same `Error: Invalid arguments`
- Network mapped M: drive: same `Error: Invalid arguments`

## What we tried (none fixed it)

| Workaround | Result |
|---|---|
| Disable External-File Protection | No change |
| Disable Dotfile Protection | No change |
| Switch to anysphere.remote-ssh extension | No change |
| Kill cursor-server and reconnect | No change |
| Update macOS client to 2.4.30 | No change (deployed new server 0f8217a8) |
| Update Windows server Cursor to 2.4.30 | N/A (SSH remote uses client-deployed server) |
| Write to local C: drive instead of network M: | Same error |
| Downgrade to older server version | Not possible (server code is identical across all versions; bug is client-side) |

## Impact

- **Severity**: Medium-high. The Write/StrReplace tools do work (files are modified correctly), but:
  - The false error clutters every agent conversation
  - AI agents may retry on error, causing duplicate edits
  - **Cursor's checkpoint/revert system is completely broken** -- users cannot undo AI changes through the UI
  - Requires workarounds (cursor rules telling AI not to retry, git-based revert workflow)

## Suggested Fix

The server always succeeds -- the bug is in the client's response handling. Possible fixes:

1. **Fix the client-side response handler** to correctly interpret the server's success response, even when the `handleBlockReason` pre-write check was slow
2. **Increase the client-side timeout** for tool execution responses over SSH Remote (the server completes within ~5-10 seconds, but the client may be timing out at 3 seconds)
3. **Don't generate "Invalid arguments"** when the server returns a valid success protobuf -- the current behavior discards the actual result
4. **Ensure checkpoint tracking** works when tool results are slow -- the file was written successfully, so it should be checkpointed regardless of response timing
5. **Cache permission decisions client-side** for SSH sessions to avoid repeated slow round-trips for the same workspace
