# Kaggle Phase A/B workflow

This is the loop for every change to the agent, from "edit a knob" to
"decide whether it shipped." It replaced an ad-hoc pile of one-off shell
and Python scripts, each named after the kernel version it was written
for (`push_kernel_v24.sh`, `watch_v23.sh`, `check_v24.py`, `sc23b.py`, ...)
and re-derived from scratch each time, because that pile is why a script
kept reinventing a broken watcher three separate times without anyone
noticing the second and third attempt shared the first attempt's bug.
The fix is `scripts/kops/`: five small scripts, checked into git,
parameterized by a label instead of a version number, extended in place
instead of copied.

## The two phases, and why they're not symmetric

- **Phase A** (`kaggle kernels push`, then Kaggle runs it): burns GPU
  quota only. Free to repeat.
- **Phase B** (clicking Submit / `kaggle competitions submit`): burns a
  **submission slot**. The CLI cannot do this for a code competition --
  it is a manual click, and it needs **explicit sign-off each time**, not
  once per session. A Phase A run that looks good is a candidate for
  Phase B, never an autopilot trigger.

Everything in `scripts/kops/` operates in Phase A. None of it submits.

## The loop

```
1. Change one thing in agent/my_agent.py or scripts/build_notebook.py,
   with a comment explaining what it's testing and why (see existing
   comments in those files for the expected density -- this project has
   been burned by a change that looked obviously good on paper twice).

2. scripts/kops/push.sh "one-line description of what this tests"
   - aborts if a submission is PENDING
   - builds notebooks/submission.ipynb
   - runs scripts/kops/checks.py against the SPLICED notebook (a commit
     deploys nothing; only the spliced output ships)
   - pushes, and appends a line to kops/history.log

3. Watch it to completion with the harness's Monitor tool, command:
       wsl bash /path/to/repo/scripts/kops/wait.sh
   Do NOT reach for `Bash --run_in_background` wrapping a sleep loop
   around a `wsl bash` call -- see "Why Monitor, not a background Bash
   loop" below.

4. scripts/kops/pull.sh <label>
   Pulls the finished run's log and server log into kaggle_runs/<label>/
   (gitignored; survives on disk across sessions, unlike a job's tmp dir).

5. python scripts/kops/analyze.py <label> [<baseline_label>]
   One report: official scorecard, server throughput, per-turn stop
   reasons. Pass a baseline label to get a diff.

6. Decide using clears and throughput, not the bare score (see "The
   scoring trap" below). Write the decision into the code comment and
   kops/history.log, whether the answer is "keep" or "revert."

7. Only on the user's explicit go-ahead, and never inferred from a score
   alone: Phase B, the Submit click.
```

## Why Monitor, not a background Bash loop

Three watchers in a row -- for v22, v23, and a v24 attempt that moved the
poll loop from inside a single long-lived `wsl bash` process to a fresh
`wsl bash` subprocess per 120s tick specifically to test whether process
lifetime was the cause -- all died with exit code 4 well before the
kernel finished. The redesign between attempt 2 and 3 didn't change the
failure, which means the cause was never process lifetime; it was
reaching for `Bash --run_in_background` at all for a "watch until done,
notify me of the outcome" task.

The harness has a dedicated tool for exactly that: **Monitor**. Point its
`command` at `scripts/kops/wait.sh` (itself: poll every 60s, print a line
only on status change, exit 0 on COMPLETE / 1 on ERROR or CANCELLED) and
it handles the polling lifecycle instead of a hand-rolled sleep loop
trying to reimplement it. If a kernel run outlives the monitor's own
timeout window, re-arm it rather than falling back to the Bash loop that
failed three times.

## The scoring trap

`level_score = min((baseline_actions/actions)^2 * 100, 115)`, summed with
weight `i` per level `i` and divided by `T = N(N+1)/2` over every level in
the game whether reached or not, averaged over 25 games. This rewards a
handful of fast/lucky clears heavily: v23 scored **0.7230** against v22's
**0.5326** while clearing only **6/25** games vs v22's **9/25**, and with
fewer total actions taken (1732 vs 2238). A higher score can mean the run
did objectively less. `scripts/kops/analyze.py` prints clears, total
actions, and total resets specifically so a score is never read alone.
See `arc-agi-3-scoring-arithmetic` in the persistent memory for the full
arithmetic.

## Security constraints (unchanged, still in force)

- Never modify or commit `.kaggle/access_token`. It's gitignored;
  `lib.sh` reads it into an env var and nothing in `scripts/kops/` prints
  it or puts it on a command line.
- `git push` runs from **Windows PowerShell**, never WSL -- WSL hangs
  silently against the Windows Credential Manager. Everything else here
  (`kaggle` CLI, pytest, `build_notebook.py`) runs under **WSL**, since
  Windows Python lacks `arcengine`.
- No `git push --force`, no history rewriting on a shared branch, no
  pushing to main/master.
- Phase B needs the user's explicit go-ahead on the specific run being
  submitted, every time. A background notification is never that
  go-ahead.

## Layout

```
scripts/kops/
  lib.sh       shared setup: cd to repo root, load the token, define
               KERNEL_ID/COMPETITION/paths. Source it, don't run it.
  push.sh      build -> verify -> push -> log to history.log
  wait.sh      poll to a terminal state; hand its command straight to
               Monitor
  pull.sh      pull a run's output into kaggle_runs/<label>/
  checks.py    guards against the SPLICED notebook; extend in place when
               a run teaches you an invariant that must hold from here on
  analyze.py   one report: scorecard + throughput + turn stats, optional
               diff against a baseline label
kops/history.log   append-only: timestamp, git SHA, one-line description
                    of what each push tested. Durable, checked into git --
                    unlike a job's scratch tmp dir, this is meant to
                    outlive the session that wrote it.
kaggle_runs/<label>/   pulled run artefacts (gitignored)
```
