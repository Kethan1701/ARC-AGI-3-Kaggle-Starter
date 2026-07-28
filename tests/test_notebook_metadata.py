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
