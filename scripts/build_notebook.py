"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

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
ACCELERATOR = "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# MODEL POLICY (ARC_POLICY)
# ─────────────────────────────────────────────────────────────────────────────
# False ships the pure-Python search that scored 0.26 -- no GPU, no weights,
# no extra mounts, nothing to provision and nothing to fail.
#
# True additionally mounts a 27B model and an offline vLLM wheelhouse and
# sets ARC_POLICY=1, which lets the model choose *which* candidate the
# search probes next. It requires a GPU accelerator, and `build()` refuses
# the combination if you forget -- the agent would otherwise spend five
# minutes failing to load 28GB of weights on a CPU box before falling back.
POLICY = False

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

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

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

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                __POLICY_ENV__python main.py --agent myagent
        """
    )
    # Substituted rather than interpolated so the CPU notebook carries no
    # trace of the policy at all -- not even an ARC_POLICY=0 that a later
    # reader might flip without mounting the weights.
    run_cell_source = run_cell_source.replace(
        "__POLICY_ENV__",
        f"ARC_POLICY=1 \\\n        ARC_POLICY_MODEL={POLICY_MODEL_DIR} \\\n        "
        if POLICY
        else "",
    )
    run_cell = code_cell(run_cell_source)

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

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    # Caught here rather than on Kaggle. A policy run without a GPU loads
    # nothing, falls back to search, and scores what the search scores --
    # while having spent a GPU-quota-free submission slot looking like a
    # model run that simply did not help.
    if POLICY and not accel["gpu"]:
        raise SystemExit(
            f"POLICY=True needs a GPU, but ACCELERATOR={ACCELERATOR!r}. "
            f"Set ACCELERATOR='rtx6000'."
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
            *([vllm_cell] if POLICY else []),
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

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
        wanted_datasets = [WHEELHOUSE] if POLICY else []
        wanted_models = [POLICY_MODEL] if POLICY else []
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
