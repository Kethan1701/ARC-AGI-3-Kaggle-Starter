"""The accelerator you ask for is not necessarily the one you get.

Three different ways to end up on a P100, none of which report anything:

1. No `machine_shape` in kernel-metadata.json. The `accelerator` key that
   `build_notebook.py` writes into *notebook* metadata is read by the web
   editor and ignored by the API, so the run takes the server default.
2. A `machine_shape` that is not the exact enum spelling. "NvidiaRtx6000"
   was accepted and silently downgraded; only "NvidiaRtxPro6000" works.
3. The right shape with the competition detached. The card is gated on
   competition membership, and passing `--accelerator` does not buy your
   way past it -- measured, see `test_the_competition_stays_attached`.

In every case the kernel still reports the shape you asked for when pulled
back, and in every case the preinstalled torch 2.10.0+cu128 cannot execute
on what you actually got (it supports sm_70-sm_120; the P100 is sm_60).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_build_notebook():
    spec = importlib.util.spec_from_file_location(
        "build_notebook_under_test", ROOT / "scripts" / "build_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_gpu_accelerator_declares_a_machine_shape():
    """Without this the API silently downgrades the run to a P100."""
    module = _load_build_notebook()
    for key, accel in module._ACCELERATORS.items():
        if accel["gpu"]:
            assert accel.get("shape"), (
                f"{key!r} enables the GPU but declares no machine_shape, so "
                f"`kaggle kernels push` would silently give it a P100"
            )
        else:
            assert accel.get("shape") is None


def test_the_rtx6000_shape_is_the_verified_enum_spelling():
    """'NvidiaRtx6000' was accepted by the API and produced a P100.
    'NvidiaRtxPro6000' produced the RTX PRO 6000 Blackwell (95GB, sm_120).
    The API validates nothing, so this string is load-bearing."""
    module = _load_build_notebook()
    assert module._ACCELERATORS["rtx6000"]["shape"] == "NvidiaRtxPro6000"


def test_shipped_metadata_agrees_with_the_selected_accelerator():
    """Whatever ACCELERATOR is set to, the committed metadata must match it --
    otherwise the next `make submit` runs on hardware nobody chose."""
    module = _load_build_notebook()
    meta = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    accel = module._ACCELERATORS[module.ACCELERATOR]

    assert meta["enable_gpu"] == accel["gpu"]
    if accel["shape"] is None:
        assert "machine_shape" not in meta
    else:
        assert meta["machine_shape"] == accel["shape"]


def test_competition_requires_internet_disabled():
    """RTX Pro 6000 sessions require it, and so does the competition."""
    meta = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    assert meta["enable_internet"] is False


def test_the_competition_stays_attached():
    """Detaching it breaks the run twice over, both times silently.

    `environment_files` is mounted from the competition, so without it
    there is no game to play. Less obviously, the RTX PRO 6000 is gated on
    competition membership. Measured with a one-cell kernel that prints
    its own device, varying one field at a time:

        competition_sources           --accelerator   device
        ["arc-prize-2026-arc-agi-3"]  absent          RTX PRO 6000 sm_120
        []                            absent          Tesla P100   sm_60
        []                            passed          Tesla P100   sm_60

    The push flag changes nothing; the attachment decides it. A detached
    kernel still reports `machine_shape: NvidiaRtxPro6000` when pulled
    back, so nothing anywhere says you were downgraded -- and on a P100
    the preinstalled torch cannot execute at all (sm_70-sm_120 vs sm_60).
    """
    meta = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    assert meta["competition_sources"] == ["arc-prize-2026-arc-agi-3"]


def test_submit_still_passes_the_shape_on_the_command_line():
    """Belt and braces, and labelled as such.

    The ablation above shows `--accelerator` is not what wins the card, so
    this pins insurance rather than the fix -- it keeps the request visible
    on the wire if Kaggle ever starts honouring it. Reading the value back
    out of kernel-metadata.json keeps one source of truth, so the flag
    cannot drift away from ACCELERATOR.
    """
    makefile = (ROOT / "Makefile").read_text()
    submit = makefile.split("submit:", 1)[1].split("\nstatus:", 1)[0]
    assert "--accelerator" in submit
    assert "machine_shape" in submit, (
        "the flag must come from kernel-metadata.json, not be hardcoded, "
        "or it will disagree with ACCELERATOR the first time it changes"
    )


# ---------------------------------------------------------------------
# The model's submission wiring
# ---------------------------------------------------------------------
def _sources(module, mode, accelerator):
    module.MODE = mode
    module.ACCELERATOR = accelerator
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in module.build()["cells"]
    )


def _plumbing(module):
    """Every cell except the one that embeds `my_agent.py`.

    The agent's own source necessarily mentions ARC_POLICY and ARC_LLM --
    that is where the flags are read -- so asking whether the *notebook*
    switches them on means asking what the surrounding cells do.
    """
    cells = []
    for cell in module.build()["cells"]:
        text = cell["source"]
        if isinstance(text, list):
            text = "".join(text)
        if text.startswith("%%writefile"):
            continue
        cells.append(text)
    return "\n".join(cells)


@pytest.mark.parametrize("mode", ["policy", "llm"])
def test_a_model_run_cannot_be_built_without_a_gpu(mode):
    """Loading 28GB of weights on a CPU box does not error -- it wastes
    five minutes, falls back to the search, and scores what the search
    scores, having burned a submission that looks like a model run."""
    module = _load_build_notebook()
    module.MODE = mode
    module.ACCELERATOR = "cpu"
    with pytest.raises(SystemExit):
        module.build()


@pytest.mark.parametrize("accelerator", ["t4", "p100"])
def test_the_llm_mode_refuses_a_card_that_cannot_run_the_weights(accelerator):
    """FP8 weights and an sm_120-only wheelhouse. A T4 is sm_75 and a
    P100 sm_60; finding that out on Kaggle costs a whole run."""
    module = _load_build_notebook()
    module.MODE = "llm"
    module.ACCELERATOR = accelerator
    with pytest.raises(SystemExit):
        module.build()


def test_an_unknown_mode_is_refused_rather_than_silently_shipping_search():
    module = _load_build_notebook()
    module.MODE = "llm-v2"
    with pytest.raises(SystemExit):
        module.build()


def test_the_policy_notebook_mounts_weights_and_vllm():
    """With internet disabled an unattached source is not a download that
    fails, it is a path that does not exist."""
    module = _load_build_notebook()
    source = _sources(module, "policy", "rtx6000")
    assert "requirements.lock" in source, "no vLLM install cell"
    assert "ARC_POLICY=1" in source, "the agent would never enable the policy"
    assert module.POLICY_MODEL_DIR in source


def test_the_llm_notebook_starts_a_server_with_tool_calling_enabled():
    """Without the tool-call parser the model can talk but never act,
    which is the entire architecture gone while the run still looks fine."""
    module = _load_build_notebook()
    source = _sources(module, "llm", "rtx6000")
    assert "vllm.entrypoints.openai.api_server" in source
    assert "--enable-auto-tool-choice" in source
    assert "qwen3_coder" in source
    assert module.POLICY_MODEL_DIR in source
    assert module.SERVED_MODEL_NAME in source


def test_the_llm_install_is_isolated_from_the_notebook_kernel():
    """A GPU run on this project has already produced no submission.parquet
    at all. Installing vLLM over the image can move numpy under the pyarrow
    that writes that file, and it fails at the last cell after nine hours."""
    module = _load_build_notebook()
    source = _sources(module, "llm", "rtx6000")
    assert f"'--target', {module.VLLM_TARGET!r}" in source or (
        "--target" in source and module.VLLM_TARGET in source
    )
    assert "--ignore-installed" in source


def test_the_llm_run_only_enables_the_agent_if_the_server_answered():
    """`/tmp/llm.env` is written after the smoke test, not after the
    launch, so a server that came up and cannot generate does not get the
    agent to depend on it."""
    module = _load_build_notebook()
    source = _sources(module, "llm", "rtx6000")
    assert "/tmp/llm.env" in source
    assert "export ARC_LLM=1" in source
    # The run command sources it conditionally: absent file, search plays.
    assert "[ -f /tmp/llm.env ]" in source


def test_a_failed_server_never_raises_and_kills_the_submission():
    """Time left on the clock is worth less than a submission that lands,
    so every failure path in the server cell must degrade, not abort."""
    module = _load_build_notebook()
    module.MODE = "llm"
    module.ACCELERATOR = "rtx6000"
    server = [
        c["source"] for c in module.build()["cells"]
        if "api_server" in str(c["source"])
    ][0]
    assert "raise" not in server, "a raise here costs the whole submission"
    assert "SystemExit" not in server


def test_save_and_run_all_actually_exercises_the_agent():
    """Phase A never plays a game, so for a long time it proved only that
    the cells parsed -- which is how a submission shipped with an
    IndexError on the action path and failed where nobody could see it."""
    module = _load_build_notebook()
    plumbing = _plumbing(module)
    assert "from my_agent import MyAgent" in plumbing
    assert "choose_action" in plumbing
    assert "SMOKE FAILED" in plumbing


def test_the_default_notebook_never_switches_a_model_on():
    """What ships today is the search that scored 0.26: no GPU, no
    weights, nothing to provision and nothing to fail."""
    module = _load_build_notebook()
    assert module.MODE == "search"
    plumbing = _plumbing(module)
    assert "ARC_POLICY" not in plumbing
    assert "api_server" not in plumbing
    assert "requirements.lock" not in plumbing
    assert "__POLICY_ENV__" not in plumbing, "placeholder left unsubstituted"


def test_the_shipped_metadata_mounts_match_the_selected_mode():
    module = _load_build_notebook()
    meta = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    if module.MODE in ("policy", "llm"):
        assert meta["dataset_sources"] == [module.WHEELHOUSE]
        assert meta["model_sources"] == [module.POLICY_MODEL]
    else:
        assert meta["dataset_sources"] == []
        assert meta["model_sources"] == []
