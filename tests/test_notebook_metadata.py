"""The accelerator you ask for is not necessarily the one you get.

`make submit` runs `kaggle kernels push` with no `--accelerator`, and the API
push path reads exactly one field:

    request.machine_shape = acc if acc else meta_data.get("machine_shape")
        -- kaggle/api/kaggle_api_extended.py

The `accelerator` key that `build_notebook.py` writes into *notebook* metadata
is read by the web editor and ignored by the API. So a GPU submission with no
`machine_shape` silently lands on the server default -- a P100 -- with no
error and no warning. Verified by probing a pushed kernel: it reported
"Tesla P100-PCIE-16GB", on which the preinstalled torch 2.10.0+cu128 cannot
run at all (it supports sm_70-sm_120; P100 is sm_60).

An unrecognised shape string fails the same silent way: "NvidiaRtx6000" was
accepted and yielded a P100. Only "NvidiaRtxPro6000" returned the real card.
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


# ---------------------------------------------------------------------
# The model policy's submission wiring
# ---------------------------------------------------------------------
def test_a_policy_run_cannot_be_built_without_a_gpu():
    """Loading 28GB of weights on a CPU box does not error -- it wastes
    five minutes, falls back to the search, and scores what the search
    scores, having burned a submission that looks like a model run."""
    module = _load_build_notebook()
    module.POLICY = True
    module.ACCELERATOR = "cpu"
    with pytest.raises(SystemExit):
        module.build()


def test_the_policy_notebook_mounts_weights_and_vllm():
    """With internet disabled an unattached source is not a download that
    fails, it is a path that does not exist."""
    module = _load_build_notebook()
    module.POLICY = True
    module.ACCELERATOR = "rtx6000"
    source = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in module.build()["cells"]
    )
    assert "requirements.lock" in source, "no vLLM install cell"
    assert "ARC_POLICY=1" in source, "the agent would never enable the policy"
    assert module.POLICY_MODEL_DIR in source


def _plumbing(module):
    """Every cell except the one that embeds `my_agent.py`.

    The agent's own source necessarily mentions ARC_POLICY -- that is
    where the flag is read -- so asking whether the *notebook* enables
    the policy means asking what the surrounding cells do.
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


def test_the_default_notebook_never_switches_the_policy_on():
    """What ships today is the search that scored 0.26: no GPU, no
    weights, nothing to provision and nothing to fail."""
    module = _load_build_notebook()
    assert module.POLICY is False
    plumbing = _plumbing(module)
    assert "ARC_POLICY" not in plumbing
    assert "requirements.lock" not in plumbing
    assert "__POLICY_ENV__" not in plumbing, "placeholder left unsubstituted"


def test_the_shipped_metadata_mounts_match_the_policy_switch():
    module = _load_build_notebook()
    meta = json.loads((ROOT / "notebooks" / "kernel-metadata.json").read_text())
    if module.POLICY:
        assert meta["dataset_sources"] == [module.WHEELHOUSE]
        assert meta["model_sources"] == [module.POLICY_MODEL]
    else:
        assert meta["dataset_sources"] == []
        assert meta["model_sources"] == []
