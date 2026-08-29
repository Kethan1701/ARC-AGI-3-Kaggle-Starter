# Poll the newest competition submission until it stops being pending.
# The Phase B watcher. Hand it to the harness's Monitor tool with
# persistent: true --
#
#   Monitor(command: "pwsh.exe -NoProfile -ExecutionPolicy Bypass -File
#                     scripts/kops/watch_submission.ps1 2>&1",
#           description: "...", persistent: true)
#
# WHY THIS IS POWERSHELL AND NOT scripts/kops/wait_submission.sh
#
# Four watchers have died with exit code 4 mid-watch (v22, v23, v24, and
# submission 55840177). Two causes were confidently recorded and both
# were wrong: "process lifetime" (falsified when v24's per-tick-subprocess
# redesign failed identically) and "reaching for Bash --run_in_background
# at all" (falsified when the prescribed replacement, the Monitor tool,
# died with the same code). A third was recorded here -- "the only
# factor common to all four is a long-lived `wsl bash` process sleeping
# between polls" -- and 2026-08-29 falsified that one too: this very
# script, PowerShell-native and never touching WSL, went silent mid-watch
# on submission 55862549 after three healthy heartbeats, with no terminal
# event and no exit code. ~4.5h passed unobserved.
#
# So: no mechanism is claimed. Three confident diagnoses have each been
# overturned by the next observation, and a fourth guess would repeat the
# pattern rather than end it. Staying out of WSL is still worth doing (it
# removes one moving part) but it is NOT a fix, and this script must not
# be trusted to run to completion.
#
# WHAT ACTUALLY HOLDS is procedural, and it has outlived every falsified
# mechanism: silence from a watcher is evidence of NOTHING. Never report
# a terminal state on a watcher's word, and re-verify with a direct read
# whenever events STOP arriving -- not only when they say something.
# A dead watcher and a healthy pending run look identical from here.
#
# .kaggle/access_token is a bearer token, so the REST API needs no venv,
# no kaggle CLI, and no WSL. The token goes into a header and nowhere
# else -- never printed, never on a command line.
#
# NOTE the status vocabulary differs from the CLI: this API returns
# lowercase 'pending' / 'complete', NOT the CLI's 'SubmissionStatus.'
# prefix. A comparison copied from wait_submission.sh will never match.
$ErrorActionPreference = 'Continue'

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tokenPath = Join-Path $repo '.kaggle\access_token'
if (-not (Test-Path $tokenPath)) {
    Write-Output "no .kaggle/access_token -- can't talk to Kaggle"
    exit 1
}
$tok = (Get-Content $tokenPath -Raw).Trim()
$h = @{ Authorization = "Bearer $tok" }
$uri = 'https://www.kaggle.com/api/v1/competitions/submissions/list/arc-prize-2026-arc-agi-3'

$last = ''
$ticks = 0
while ($true) {
    $rows = $null
    try {
        $rows = Invoke-RestMethod -Uri $uri -Headers $h -Method Get -TimeoutSec 60
    } catch {
        # Retry a transient network/API failure rather than dying. A
        # watcher that exits on a blip is indistinguishable from one that
        # exits on a real terminal state -- that ambiguity is exactly how
        # wait_submission.sh's CSV bug produced a confident false
        # "resolved" on its very first poll.
        Start-Sleep -Seconds 120
        continue
    }
    if (-not $rows -or $rows.Count -eq 0) { Start-Sleep -Seconds 120; continue }
    $r = $rows[0]
    $st = [string]$r.status
    if ([string]::IsNullOrWhiteSpace($st)) { Start-Sleep -Seconds 120; continue }

    if ($st -ne $last) {
        "{0}  ref={1}  status={2}  score={3}" -f `
            (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $r.ref, $st, $r.publicScore
        $last = $st
    }
    if ($st -ne 'pending') {
        "no longer pending."
        exit 0
    }

    $ticks++
    # Heartbeat every ~30 min. Without it a live watcher and a dead one
    # look identical from the outside, which is how three of the four
    # exit-4 deaths went unnoticed for hours.
    if ($ticks % 15 -eq 0) {
        "{0}  heartbeat: still pending ({1} polls)" -f `
            (Get-Date).ToUniversalTime().ToString('HH:mm:ss'), $ticks
    }
    Start-Sleep -Seconds 120
}
