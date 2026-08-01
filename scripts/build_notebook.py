"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /tmp/ — its body is THIS file.
  Cell 3: (MODE="llm" only) stand up a vLLM OpenAI server on the mounted
          weights and, only once it answers a real chat completion, write
          /tmp/llm.env so the agent is told to use it.
  Cell 4: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 5: otherwise (during commit / save-and-run-all), drive the agent
          through a handful of turns against a synthetic frame. The rerun
          is the only place a game is actually played, so without this
          Phase A proves nothing but that the cells parsed — which is how a
          submission once shipped with an IndexError on the action path.
  Cell 6: and write a dummy submission.parquet so the commit succeeds. The
          real one is produced by the gateway during the rerun.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
#
# "cpu" is correct for the current agent: it is a pure-Python graph search
# with no model, so a GPU would sit idle while still consuming GPU quota
# and adding a provisioning failure mode to every run. Switch this only if
# you actually wire in local model weights.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "rtx6000"

# ─────────────────────────────────────────────────────────────────────────────
# WHAT PLAYS THE GAMES
# ─────────────────────────────────────────────────────────────────────────────
#   "search"  the pure-Python state-graph search that scored 0.26. No GPU, no
#             weights, no extra mounts: nothing to provision and nothing to
#             fail. This is the fallback in every other mode too.
#
#   "policy"  search, but a model ranks which candidate to probe next
#             (ARC_POLICY). Loads weights in-process via the vLLM Python API.
#             Superseded by "llm" and kept only so the comparison is still
#             runnable -- it never beat the search by enough to matter.
#
#   "llm"     the model plays. It is given a Python sandbox and an object
#             view of the board and writes its own search, executing action
#             sequences from inside its own code (ARC_LLM). This is the
#             architecture the 1.0+ leaderboard cluster runs, and the reason
#             the split is 0.26-0.54 for search-only against 1.21-1.47 for
#             a model with tools. Needs the GPU, the weights and the
#             wheelhouse, and stands up a vLLM OpenAI server as a subprocess.
#
# `build()` refuses a GPU mode on a CPU accelerator: the agent would
# otherwise spend minutes failing to load 28GB of weights before falling
# back, having spent a submission slot to score what the search scores.
MODE = "llm"

_MODES = ("search", "policy", "llm")

# vLLM is not on the competition image and the run has no internet, so it
# comes from a mounted wheelhouse. Despite the "h100" in the name, this
# one's vllm 0.19.0 wheel carries sm_120 kernels and runs on the Blackwell
# RTX PRO 6000 -- verified by probe, not assumed from the title.
WHEELHOUSE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"

# Kaggle model slug, and the path it mounts at. ARC-AGI-3 nests inputs
# under /kaggle/input/models/... rather than the flat /kaggle/input/<slug>
# used elsewhere on Kaggle; assuming the flat layout cost a probe cycle.
POLICY_MODEL = "michaelpoluektov/qwen3-6-27b-fp8/transformers/default/1"
POLICY_MODEL_DIR = f"/kaggle/input/models/{POLICY_MODEL}"

# --- vLLM server, for MODE="llm" -------------------------------------------
# The name the served model answers to. It is an alias, not a lookup: the
# weights come from the mounted directory above and this only has to match
# what the agent puts in `"model"`, which is ARC_LLM_MODEL's default.
SERVED_MODEL_NAME = "vrfai/Qwen3.6-27B-FP8"
SERVER_PORT = 1234

# vLLM is installed into its OWN directory rather than over the image.
#
# This is the fix for a failure mode already seen on this project: a GPU
# run that produced no submission.parquet at all. Installing vLLM on top
# of the competition image drags in its own torch/numpy pins, and the
# thing that writes submission.parquet is pandas -> pyarrow -> numpy. A
# resolver that upgrades numpy under a pyarrow already imported does not
# fail at install time; it fails at the last cell, after nine hours, with
# nothing to submit. An isolated --target cannot do that: the server
# subprocess gets it on PYTHONPATH and the notebook kernel never sees it.
#
# /tmp, NOT /kaggle/working. Anything under /kaggle/working is a notebook
# OUTPUT: the first run put several GB of wheels there, which the "Submit
# to Competition" file picker then offers alongside submission.parquet --
# the same trap that keeps my_agent.py in /tmp -- and which counts against
# the output size limit for nothing.
VLLM_TARGET = "/tmp/vllm-site-packages"

# Long because it is honest. Loading 28GB of FP8 weights and compiling
# CUDA graphs is minutes, not seconds, and a wait that expires early
# silently demotes the whole run to the search.
SERVER_WAIT_SECONDS = 900

# --- the Phase A rehearsal --------------------------------------------------
# One game, played for real. ls20 is the pick because its per-level
# baselines are known ([22, 123, 73, 84, 96, 192, 186]) and it is
# movement-driven, so "did the model make progress" is legible in the
# actions/baseline column rather than needing interpretation.
# ls20 is first for continuity: v16, v17 and v18 all played it, so its
# numbers are directly comparable and every diagnosis so far came from it.
# vc33 is the contrast -- a 7-action level-1 baseline against ls20's 22 --
# which asks a different question. ls20 tests whether the pipeline can find
# a goal at all; vc33 tests whether it can be CHEAP, and cheap is the only
# thing the quadratic actually pays for.
SMOKE_GAMES = ["ls20", "vc33"]

# This was 300, and 300 invalidated three GPU runs.
#
# The claim it carried -- "enough to cross level 1" -- was never checked.
# It is false: our own search needs 526 actions to clear ls20 level 1, and
# at 300 steps the search clears nothing either. So v13, v14 and v15 all
# stopped at 304/289/317 actions and were read as "the model cannot clear
# a level", when what they actually showed was a cap being hit. Three runs
# and about eight GPU hours went into answering a question the harness had
# already decided.
#
# The bound that matters is wall clock, not steps, so the step cap is now
# set well clear of anything we might legitimately hit and the rehearsal is
# limited by ARC_RUN_BUDGET_SECONDS below instead. 1200 leaves room to
# clear level 1 (baseline 22) and get a real look at level 2 (baseline 123).
SMOKE_STEPS = 1200

# The real bound on the rehearsal. The agent reads this into `_DEADLINE`
# and stops on its own, so a slow turn shortens the run instead of
# overrunning Save & Run All. Forty minutes on top of ~6 minutes of server
# startup keeps the whole notebook comfortably inside its limit.
#
# v16 spent all of 2400s and got 7 turns and 140 actions out of it -- it
# died on this budget, not on SMOKE_STEPS, with verify accuracy still
# climbing when the clock ran out. An hour buys the turns it needed to
# finish converging. It costs nothing in the real run: there the 25 games
# are played concurrently in one process, so an hour per game is normal
# rather than 25 hours of work.
# Now PER GAME, not for the whole rehearsal. Two games at half an hour
# each costs the same wall clock as v18's single hour and samples twice as
# much. Half an hour bought v18 roughly three turns, which is enough to
# see the pipeline reach the planning stage but not enough to watch it
# converge -- that is the deliberate trade for a second game.
SMOKE_SECONDS = 1800

# Internal mapping; don't edit unless Kaggle adds new options.
#
# "name" goes in the *notebook* metadata, which is what the web editor reads.
# "shape" goes in kernel-metadata.json as `machine_shape`, which is the ONLY
# channel the API push path honours:
#
#     request.machine_shape = acc if acc else meta_data.get("machine_shape")
#         -- kaggle/api/kaggle_api_extended.py
#
# `make submit` runs `kaggle kernels push` with no --accelerator, so without
# `machine_shape` the notebook metadata is ignored and `enable_gpu: true`
# silently falls back to the server default: a P100. That is not an error and
# emits no warning -- verified by probing a pushed kernel, which reported
# "Tesla P100-PCIE-16GB". On a P100 the preinstalled torch 2.10.0+cu128 is
# unusable anyway; it supports sm_70-sm_120 and P100 is sm_60.
#
# An unrecognised shape is likewise accepted silently and yields a P100, so
# these strings must match Kaggle's enum exactly -- "NvidiaRtx6000" is wrong
# and was swallowed. "NvidiaRtxPro6000" is verified: it returned "NVIDIA RTX
# PRO 6000 Blackwell Server Edition", 95GB VRAM, sm_120.
#
# The shape is necessary but NOT sufficient: the card is gated on competition
# membership too. Measured with a one-cell kernel that prints its own device,
# holding every other field constant (see tests/test_notebook_metadata.py):
#
#   competition_sources         --accelerator     device
#   ["arc-prize-2026-arc-agi-3"]  absent          RTX PRO 6000  sm_120
#   []                            absent          Tesla P100    sm_60
#   []                            passed          Tesla P100    sm_60
#
# So the push flag changes nothing and the attachment decides it. A kernel
# with the right shape and no competition is still handed a P100, and pulling
# it back still reports `machine_shape: NvidiaRtxPro6000` -- Kaggle records
# what you asked for and grants what you are entitled to, silently.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False, "shape": None},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True,  "shape": "NvidiaTeslaT4"},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True,  "shape": "NvidiaTeslaP100"},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True,  "shape": "NvidiaRtxPro6000"},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    # Only the competition's own offline wheelhouse is needed. `my_agent.py`
    # is deliberately self-contained (standard library + arcengine + the
    # framework's Agent base class), so there is no private dataset to attach
    # and no second install step. Two separate Kaggle failures came out of
    # that plumbing — a kaggle.com URL that cannot resolve with internet
    # disabled, then a follow-up edit that dropped the arc-agi install
    # altogether — and removing it deletes the whole failure mode.
    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # Installed from a mount, offline. The wheelhouse directory is located
    # by search rather than by a hardcoded path: the first attempt at this
    # assumed the flat /kaggle/input/<slug> layout used elsewhere on
    # Kaggle, and ARC-AGI-3 nests under /kaggle/input/datasets/<owner>/.
    # It reports loudly on failure -- a silently absent vLLM would look
    # exactly like a model that merely declined to answer, and the agent
    # would fall back to search and score 0.26 while appearing to work.
    vllm_cell = code_cell(
        dedent(
            f"""\
            import pathlib, subprocess, sys

            lock = next(
                pathlib.Path('/kaggle/input').rglob('requirements.lock'), None)
            if lock is None:
                print('WHEELHOUSE NOT FOUND -- is {WHEELHOUSE} attached?')
                for p in sorted(pathlib.Path('/kaggle/input').glob('*/*')):
                    print('  ', p)
            else:
                print('wheelhouse:', lock.parent)
                rc = subprocess.call([
                    sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--find-links', str(lock.parent),
                    '--requirement', str(lock),
                ])
                print('pip rc:', rc)
                import importlib
                try:
                    v = importlib.import_module('vllm')
                    print('vllm', v.__version__, 'ready')
                except Exception as exc:
                    print('vllm UNUSABLE:', exc)
            """
        )
    )

    # Stand up a vLLM OpenAI server for the agent to talk to.
    #
    # Nothing here raises. Every failure prints loudly and leaves
    # /tmp/llm.env unwritten, which means ARC_LLM is never set and the
    # agent plays the search -- a worse score, but a score. Raising would
    # fail the cell and produce no submission at all, and time left on the
    # clock is worth less than a submission that lands.
    server_cell = code_cell(
        dedent(
            f"""\
            import json, os, pathlib, subprocess, sys, time, urllib.request

            MODEL_DIR = {POLICY_MODEL_DIR!r}
            SERVED    = {SERVED_MODEL_NAME!r}
            TARGET    = {VLLM_TARGET!r}
            PORT      = {SERVER_PORT}
            BASE      = f'http://127.0.0.1:{{PORT}}/v1'

            def bail(msg):
                print('LLM DISABLED:', msg)
                print('The search will play every game. Expect ~0.26.')
                return False

            def setup():
                gpu = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True).stdout.strip()
                print('GPU:', gpu or '(nvidia-smi said nothing)')
                # Checked, not assumed. The RTX PRO 6000 is granted on
                # competition entitlement, not on what the metadata asked
                # for, and a silent P100 substitution has happened here
                # before. The wheelhouse carries sm_120 kernels only.
                if 'RTX PRO 6000' not in gpu:
                    return bail(f'expected an RTX PRO 6000, got {{gpu!r}}')

                if not pathlib.Path(MODEL_DIR).is_dir():
                    return bail(f'weights not mounted at {{MODEL_DIR}}')

                lock = next(
                    pathlib.Path('/kaggle/input').rglob('requirements.lock'), None)
                if lock is None:
                    for p in sorted(pathlib.Path('/kaggle/input').glob('*/*')):
                        print('  mounted:', p)
                    return bail('vLLM wheelhouse not attached')

                print('wheelhouse:', lock.parent)
                rc = subprocess.call([
                    sys.executable, '-m', 'pip', 'install', '--no-index',
                    '--find-links', str(lock.parent), '--requirement', str(lock),
                    '--target', TARGET,
                    # Isolated and self-contained: --ignore-installed so it
                    # does not decide the image's copy will do, --only-binary
                    # so nothing tries to compile without internet, and
                    # --no-compile because .pyc for 28GB of deps is minutes
                    # of wall clock for nothing.
                    '--ignore-installed', '--only-binary', ':all:', '--no-compile',
                ])
                if rc != 0:
                    return bail(f'pip install failed (rc={{rc}})')

                env = dict(os.environ)
                env['PYTHONPATH'] = TARGET + ':' + env.get('PYTHONPATH', '')
                env['VLLM_NO_USAGE_STATS'] = '1'
                env['HF_HUB_OFFLINE'] = '1'
                env['TRANSFORMERS_OFFLINE'] = '1'

                log = open('/kaggle/working/vllm-server.log', 'wb')
                proc = subprocess.Popen([
                    sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                    '--model', MODEL_DIR,
                    '--served-model-name', SERVED,
                    '--host', '127.0.0.1', '--port', str(PORT),
                    '--tensor-parallel-size', '1',
                    # The three flags the whole design rests on: tool calls
                    # have to be parsed out of Qwen's format, or the model
                    # can talk but never act.
                    '--enable-auto-tool-choice',
                    '--tool-call-parser', 'qwen3_coder',
                    '--reasoning-parser', 'qwen3',
                    '--generation-config', 'vllm',
                    '--enable-prefix-caching',
                    '--default-chat-template-kwargs',
                    json.dumps({{'preserve_thinking': True}}),
                    '--max-model-len', '65536',
                    '--gpu-memory-utilization', '0.92',
                ], env=env, stdout=log, stderr=subprocess.STDOUT)

                deadline = time.time() + {SERVER_WAIT_SECONDS}
                while time.time() < deadline:
                    if proc.poll() is not None:
                        print(open('/kaggle/working/vllm-server.log').read()[-4000:])
                        return bail(f'server exited with {{proc.returncode}}')
                    try:
                        urllib.request.urlopen(BASE + '/models', timeout=5).read()
                        break
                    except Exception:
                        time.sleep(5)
                else:
                    return bail('server never came up')
                print(f'server up after {{int(time.time() - deadline + {SERVER_WAIT_SECONDS})}}s')

                # Prove it answers a real chat completion with a tool
                # declared. /v1/models responds well before the model can
                # generate, and "up" is not the property we need.
                body = json.dumps({{
                    'model': SERVED,
                    'messages': [{{'role': 'user', 'content': 'Reply with OK.'}}],
                    'max_tokens': 8, 'temperature': 0.0,
                }}).encode()
                try:
                    req = urllib.request.Request(
                        BASE + '/chat/completions', data=body,
                        headers={{'Content-Type': 'application/json'}})
                    reply = json.loads(urllib.request.urlopen(req, timeout=180).read())
                    print('smoke test:', reply['choices'][0]['message'])
                except Exception as exc:
                    return bail(f'smoke test failed: {{type(exc).__name__}}: {{exc}}')
                return True

            if setup():
                # Only now is the agent told to depend on it.
                with open('/tmp/llm.env', 'w') as f:
                    f.write(f'export ARC_LLM=1\\n')
                    f.write(f'export ARC_LLM_BASE_URL={{BASE}}\\n')
                    f.write(f'export ARC_LLM_MODEL={{SERVED}}\\n')
                print('LLM ENABLED at', BASE)
            """
        )
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    # The repo's own scorer, shipped verbatim. Its three path assumptions
    # are env-overridable precisely so this works, which keeps Phase A
    # honest: it measures with the same harness the repo measures with,
    # instead of a notebook-only imitation that could agree with nothing.
    scorer_src = (ROOT / "scripts" / "score_competition.py").read_text()
    write_scorer_cell = code_cell(
        "%%writefile /tmp/score_competition.py\n" + scorer_src
    )

    # Framework setup, UNCONDITIONALLY.
    #
    # This used to live inside the rerun branch, which meant `agents` did
    # not exist during Save & Run All -- so the Phase A smoke test could
    # not import the agent at all and died on `ModuleNotFoundError: No
    # module named 'agents'`. The framework is what `my_agent.py` derives
    # its base class from, so anything that imports the agent needs it,
    # not just the run.
    framework_cell = code_cell(
        dedent(
            """\
            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
            from dotenv import load_dotenv
            from .agent import Agent, Playback
            from .swarm import Swarm
            from .templates.random_agent import Random
            from .templates.my_agent import MyAgent

            load_dotenv()

            AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
                'random': Random,
                'myagent': MyAgent,
            }
            \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
            HOST=gateway
            PORT=8001
            ARC_API_KEY=test-key-123
            ARC_BASE_URL=http://gateway:8001/
            OPERATION_MODE=online
            ENVIRONMENTS_DIR=
            RECORDINGS_DIR=/kaggle/working/server_recording
            \"\"\")
            print('framework ready')
            """
        )
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Run it. The gateway records every action and emits submission.parquet.
            #
            # /tmp/llm.env exists only if the vLLM server actually came up AND
            # answered a chat completion. When it does not, ARC_LLM stays unset
            # and the agent plays the search. That is the entire fallback, and
            # it is one line of shell on purpose: there is no state to get
            # wrong and nothing to misreport.
            !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg \\
                __POLICY_ENV__bash -c '[ -f /tmp/llm.env ] && . /tmp/llm.env; \\
                exec python main.py --agent myagent'
        """
    )
    # Substituted rather than interpolated so the CPU notebook carries no
    # trace of the policy at all -- not even an ARC_POLICY=0 that a later
    # reader might flip without mounting the weights.
    run_cell_source = run_cell_source.replace(
        "__POLICY_ENV__",
        f"ARC_POLICY=1 \\\n        ARC_POLICY_MODEL={POLICY_MODEL_DIR} \\\n        "
        if MODE == "policy"
        else "",
    )
    run_cell = code_cell(run_cell_source)

    # Phase A ("Save & Run All") never plays a game -- the gateway sidecar
    # only exists during the competition rerun -- so for a long time it
    # proved nothing except that the notebook's cells parsed. That is how
    # submission 55099108 shipped with an IndexError on the action path and
    # failed where it could not be seen.
    #
    # This does not need the gateway sidecar, because the competition
    # dataset ships `environment_files/` -- all 25 real games, offline. So
    # Phase A stands up the genuine arc_agi Flask app with
    # competition_mode=True and plays a REAL game under the real rules,
    # using the same `score_competition.py` the repo scores with rather
    # than a second, less honest harness written for the notebook.
    #
    # The first version of this cell played a synthetic all-zero frame.
    # That proved the loop round-trips and nothing about whether the model
    # can play: it reported 136 actions with no way to tell exploration
    # from thrashing, because a board that never changes cannot be solved.
    smoke_cell = code_cell(
        dedent(
            f"""\
            import os, subprocess, sys

            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                COMP = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
                env = dict(os.environ)
                env['ARC_FRAMEWORK_DIR'] = '/kaggle/working/ARC-AGI-3-Agents'
                env['ARC_ENVIRONMENTS_DIR'] = COMP + '/environment_files'
                env['ARC_AGENT_PATH'] = '/tmp/my_agent.py'
                env['RECORDINGS_DIR'] = '/tmp/recordings'
                env['MPLBACKEND'] = 'agg'

                # Whatever the server cell proved. The agent reads ARC_LLM at
                # import time, and the subprocess inherits this env, so
                # without it Phase A would exercise the search and tell us
                # nothing about the path being shipped.
                if os.path.exists('/tmp/llm.env'):
                    for line in open('/tmp/llm.env'):
                        line = line.strip().replace('export ', '', 1)
                        if '=' in line:
                            k, _, v = line.partition('=')
                            env[k] = v
                print('ARC_LLM =', env.get('ARC_LLM', '0'))

                # Bound the rehearsal by time, not by the step cap. See
                # SMOKE_STEPS in build_notebook.py for why that distinction
                # cost three runs.
                #
                # One subprocess PER GAME, because the agent computes its
                # deadline once at import: a single process playing two
                # games serially gives the whole budget to the first, which
                # is what v18 did -- 3806 seconds, one game, and 24 games
                # that never started.
                rcs = []
                for game in {SMOKE_GAMES!r}:
                    env['ARC_RUN_BUDGET_SECONDS'] = str({SMOKE_SECONDS})
                    print(f'=== rehearsing {{game}} '
                          f'({SMOKE_SECONDS}s budget) ===', flush=True)
                    rcs.append(subprocess.call(
                        [sys.executable, '/tmp/score_competition.py',
                         '--game', game, '--max-steps', str({SMOKE_STEPS})],
                        env=env))

                # A non-zero exit is reported, not raised. This cell must not
                # be able to fail the commit -- the run still has to produce
                # a submission.parquet below.
                rc = max(rcs) if rcs else 0
                print('SMOKE OK' if rc == 0 else
                      f'SMOKE FAILED (rc={{rc}}) -- this agent would error on '
                      f'the leaderboard. Fix before submitting.')
            """
        )
    )

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if MODE not in _MODES:
        raise SystemExit(f"Unknown MODE={MODE!r}. Pick one of: {list(_MODES)}")
    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    # Caught here rather than on Kaggle. A model run without a GPU loads
    # nothing, falls back to search, and scores what the search scores --
    # while having spent a submission slot looking like a model run that
    # simply did not help.
    if MODE != "search" and not accel["gpu"]:
        raise SystemExit(
            f"MODE={MODE!r} needs a GPU, but ACCELERATOR={ACCELERATOR!r}. "
            f"Set ACCELERATOR='rtx6000'."
        )
    # The weights are FP8 and the wheelhouse ships sm_120 kernels only.
    # A T4 (sm_75) or P100 (sm_60) cannot run either, and finding that out
    # on Kaggle costs a run.
    if MODE == "llm" and ACCELERATOR != "rtx6000":
        raise SystemExit(
            f"MODE='llm' needs the Blackwell card: FP8 weights and sm_120 "
            f"kernels. Set ACCELERATOR='rtx6000', not {ACCELERATOR!r}."
        )

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            *([vllm_cell] if MODE == "policy" else []),
            write_agent_cell,
            write_scorer_cell,
            # Unconditional: the smoke test imports the agent, and the
            # agent's base class lives in this package.
            framework_cell,
            # The server cell comes AFTER the agent is written, so the
            # smoke test below can exercise the agent against a live
            # server during Save & Run All rather than only in the rerun.
            *([server_cell] if MODE == "llm" else []),
            run_cell,
            smoke_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(mode: {MODE}, accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU <-> GPU. `machine_shape` matters as much as
    # `enable_gpu`: it is the only field the API push path reads, and getting
    # it wrong costs you a P100 with no error. See _ACCELERATORS.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        accel = _ACCELERATORS[ACCELERATOR]
        changed = []

        if meta.get("enable_gpu") != accel["gpu"]:
            meta["enable_gpu"] = accel["gpu"]
            changed.append(f"enable_gpu={accel['gpu']}")

        # Drop the key entirely for CPU rather than writing null -- the API
        # treats a missing shape and a null shape the same, and an explicit
        # null invites the reader to think a shape was requested.
        if accel["shape"] is None:
            if meta.pop("machine_shape", None) is not None:
                changed.append("machine_shape removed")
        elif meta.get("machine_shape") != accel["shape"]:
            meta["machine_shape"] = accel["shape"]
            changed.append(f"machine_shape={accel['shape']}")

        # The weights and the vLLM wheelhouse have to be *attached*, not
        # merely referenced: with internet disabled an unmounted source is
        # not a download that fails, it is a path that does not exist.
        needs_model = MODE in ("policy", "llm")
        wanted_datasets = [WHEELHOUSE] if needs_model else []
        wanted_models = [POLICY_MODEL] if needs_model else []
        if meta.get("dataset_sources") != wanted_datasets:
            meta["dataset_sources"] = wanted_datasets
            changed.append(f"dataset_sources={wanted_datasets}")
        if meta.get("model_sources") != wanted_models:
            meta["model_sources"] = wanted_models
            changed.append(f"model_sources={wanted_models}")

        if changed:
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced {', '.join(changed)} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
