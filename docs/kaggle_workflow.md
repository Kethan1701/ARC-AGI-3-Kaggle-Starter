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
- **Phase B** (`kaggle competitions submit`, or the Submit button on the
  kernel page): burns a **submission slot**, and needs **explicit
  sign-off each time**, not once per session. A Phase A run that looks
  good is a candidate for Phase B, never an autopilot trigger.

  **Corrected 2026-08-28.** This bullet used to say "The CLI cannot do
  this for a code competition -- it is a manual click." That is false.
  `kaggle competitions submit` has `-k` and `-v` precisely for code
  competitions:

      kaggle competitions submit arc-prize-2026-arc-agi-3 \
          -k kethansaikoneru/arc-prize-2026-arc-agi-3-starter \
          -v <kernel-version> -f submission.parquet \
          -m "<what this run tests>"

  Verified against submission 55840177. Two traps: the version must be a
  kernel version that reached COMPLETE (a cancelled run has no output to
  submit), and the success message is only "N submissions remaining
  today" -- which reads exactly like a refusal when N is 0. Confirm with
  `kaggle competitions submissions` instead of trusting that line.

  That the CLI *can* submit makes the sign-off rule more important, not
  less: the manual click used to be an accidental safety interlock, and
  it never really was one. Nothing but the rule now stands between a
  script and a spent slot.

Nothing in `scripts/kops/` submits. `wait_submission.sh` observes Phase B
but cannot start it; everything else operates in Phase A.

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

3. Watch it to completion with the harness's Monitor tool, armed
   `persistent: true`. A Phase A rehearsal runs ~2h; Monitor's
   non-persistent timeout maxes at 1h, so without that flag the watcher
   is guaranteed to die before the kernel finishes.
   Do NOT put a long-lived `wsl bash` sleep loop behind it -- that is the
   one thing all four exit-4 watcher deaths share. See "The exit-4
   watcher failure" below.

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
   alone: Phase B.
       kaggle competitions submit <comp> -k <owner/kernel> \
           -v <kernel-version> -f submission.parquet -m "<what it tests>"
   Then watch it with the Monitor tool, `persistent: true`, driving a
   PowerShell-native poll of the Kaggle REST API rather than
   `wsl bash wait_submission.sh` -- see "The exit-4 watcher failure".
   Phase B replays the notebook against the HIDDEN game set, on the
   agent's real 7.5h budget -- expect hours, not the ~2h of a Phase A
   rehearsal. Budget the watch accordingly: 55840177 was still `pending`
   6h23m after submission, which is normal, not a stall.
```

## The exit-4 watcher failure

**Rewritten 2026-08-28. The previous version of this section named the
wrong cause, and acting on it cost a fourth dead watcher.**

It used to read: three watchers (v22, v23, a v24 attempt) all died with
exit code 4 before their kernel finished, the redesign between attempts
didn't change it, therefore the cause was "reaching for
`Bash --run_in_background` at all," and the fix was the harness's
**Monitor** tool.

That prescription was followed for Phase B submission 55840177. The
Monitor-driven watcher printed one status line at 09:29:28 and **died
with exit code 4** -- the same code, under the tool that was supposed to
be the cure. So `Bash --run_in_background` was never the cause. Four
failures, and the only thing all four share is a **long-lived
`wsl bash` process sleeping between polls**. The harness tool wrapping
it was never the variable.

The mechanism inside WSL is still unproven -- do not write one down here
until it is measured. What is established is narrower and enough to act
on: keep the watch process out of WSL.

Two things to do instead, both independent of the unproven mechanism:

1. **Poll from PowerShell, not WSL.** `.kaggle/access_token` is a bearer
   token, so the Kaggle REST API is reachable with `Invoke-RestMethod`
   and no venv, no `kaggle` CLI, and no WSL:

       GET https://www.kaggle.com/api/v1/competitions/submissions/list/<comp>
       Authorization: Bearer <token>

   Returns JSON with `ref`, `status` (lowercase `pending`/`complete`
   here, *not* the CLI's `SubmissionStatus.` prefix -- a watcher ported
   between the two must change its comparison or it will never match),
   and `publicScore`. Read the token into a variable and put it only in a
   header; never on a command line.

2. **Set `persistent: true`.** This is a separate, certain bug in the old
   advice. Monitor's non-persistent timeout maxes out at one hour, and
   Phase B replays against the hidden game set on the agent's real 7.5h
   budget. A non-persistent watcher **cannot** survive to a Phase B
   score no matter what causes exit 4. "Re-arm it if the run outlives the
   window" was busywork papering over a wrong flag.

The `wsl bash` versions of `wait.sh` / `wait_submission.sh` are still
correct as **one-shot logic** and still fine for short Phase A polls, but
prefer `check_kernel.sh` / `check_submission.sh` for a direct read and a
PowerShell-native loop for anything that has to stay alive for hours.

**Standing rule, unchanged and reinforced:** never report a terminal
state from a watcher's silence or its exit alone. Confirm with an
independent direct status read first. Three of these four failures were
initially indistinguishable from "still running," and the CSV-parsing bug
in `wait_submission.sh` produced the opposite error -- a confident false
"resolved" on the first poll.

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
  wait.sh      poll a KERNEL to a terminal state; hand its command
               straight to Monitor
  watch_submission.ps1
               THE Phase B watcher. PowerShell + Kaggle REST API, no WSL,
               for Monitor with persistent: true. Use this, not
               wait_submission.sh -- see "The exit-4 watcher failure"
  wait_submission.sh
               the WSL/CLI predecessor. Logic is correct and it is fine
               for a short poll, but as a long-lived `wsl bash` process it
               is the shape that has died with exit code 4 four times
  check_kernel.sh / check_submission.sh
               one-shot status reads, for when you want an answer now
               rather than a watch
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
