# Poll the kernel to a terminal state. The Phase A counterpart to
# watch_submission.ps1, and PowerShell for the same reason: four watchers
# have died with exit code 4 and the only thing all four shared was a
# long-lived `wsl bash` process sleeping between polls. See
# docs/kaggle_workflow.md, "The exit-4 watcher failure".
#
# Hand to Monitor with persistent: true --
#
#   Monitor(command: "pwsh.exe -NoProfile -ExecutionPolicy Bypass -File
#                     scripts/kops/watch_kernel.ps1 2>&1",
#           description: "...", persistent: true)
#
# persistent matters: Monitor's non-persistent timeout caps at 1h and a
# rehearsal runs SMOKE_SECONDS (7200s) plus queue time, so a non-persistent
# watch cannot reach the end of a normal run.
#
# The token goes into a header and nowhere else -- never printed, never on
# a command line.
$ErrorActionPreference = 'Continue'

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tokenPath = Join-Path $repo '.kaggle\access_token'
if (-not (Test-Path $tokenPath)) {
    Write-Output "no .kaggle/access_token -- can't talk to Kaggle"
    exit 1
}
$tok = (Get-Content $tokenPath -Raw).Trim()
$h = @{ Authorization = "Bearer $tok" }
$uri = 'https://www.kaggle.com/api/v1/kernels/status' +
       '?userName=kethansaikoneru&kernelSlug=arc-prize-2026-arc-agi-3-starter'

# Anything not in this set is treated as terminal. Listing the RUNNING
# states rather than the finished ones is deliberate: a status this script
# has never seen should end the watch loudly, not be silently polled
# forever. Silence is the failure mode that cost this project three
# unnoticed dead watchers.
$live = @('running', 'queued', 'pending')

$last = ''
$ticks = 0
while ($true) {
    $r = $null
    try {
        $r = Invoke-RestMethod -Uri $uri -Headers $h -Method Get -TimeoutSec 60
    } catch {
        # Transient failure: retry rather than die. A watcher that exits on
        # a blip is indistinguishable from one that exits on a real result.
        Start-Sleep -Seconds 60
        continue
    }
    $st = [string]$r.status
    if ([string]::IsNullOrWhiteSpace($st)) { Start-Sleep -Seconds 60; continue }

    if ($st -ne $last) {
        "{0}  status={1}" -f (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $st
        $last = $st
    }

    if ($live -notcontains $st.ToLower()) {
        if ($r.failureMessage) { "failure: $($r.failureMessage)" }
        "kernel reached a terminal state: $st"
        # Exit 0 on a clean finish, 1 otherwise, so the exit code alone
        # distinguishes them without re-reading the stream.
        if ($st.ToLower() -eq 'complete') { exit 0 } else { exit 1 }
    }

    $ticks++
    # Heartbeat every ~30 min: a live watcher and a dead one must not look
    # the same from outside.
    if ($ticks % 30 -eq 0) {
        "{0}  heartbeat: still {1} ({2} polls)" -f `
            (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $st, $ticks
    }
    Start-Sleep -Seconds 60
}
