"""GPU integration test (skipped unless ISHINCA_CKPT and ISHINCA_INPUTS_NPZ are set).

  ISHINCA_CKPT=./Ishinca-4B ISHINCA_INPUTS_NPZ=inputs.npz python -m pytest tests/test_rollout_parity.py

Checks on one clip: (1) the explicit Ishinca rollout equals the upstream HF-generate path, (2) DART keep=1.0 equals no DART.
"""

import os

import numpy as np
import pytest
import torch

CKPT, NPZ = os.environ.get("ISHINCA_CKPT"), os.environ.get("ISHINCA_INPUTS_NPZ")
pytestmark = pytest.mark.skipif(not (CKPT and NPZ and torch.cuda.is_available()), reason="needs checkpoint, inputs, GPU")


@pytest.fixture(scope="module")
def setup():
    from ishinca import build_model_inputs, load_ishinca
    from ishinca.data import load_inputs
    from ishinca.inputs import get_processor

    model = load_ishinca(CKPT, dart_keep=None)
    d = load_inputs(NPZ)
    inputs = build_model_inputs(model, get_processor(model), d["image_frames"], d["camera_indices"],
                                d["ego_history_xyz"], d["ego_history_rot"])
    return model, inputs


def _run(model, inputs, **kw):
    torch.manual_seed(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=inputs, noise_seed=7, return_extra=True, **kw)
    return xyz[0, 0, 0, :, :2].float().cpu().numpy(), str(extra["cot"][0, 0, 0])


def test_rollout_matches_upstream(setup):
    model, inputs = setup
    model.set_dart(None)
    (a, ca), (b, cb) = _run(model, inputs), _run(model, inputs, force_ishinca_rollout=True)
    assert ca == cb
    assert np.abs(a - b).max() < 1e-3


def test_dart_keep_all_is_identity(setup):
    model, inputs = setup
    model.set_dart(None)
    a, ca = _run(model, inputs, force_ishinca_rollout=True)
    model.set_dart(1.0)
    try:
        b, cb = _run(model, inputs)
    finally:
        model.set_dart(None)
    assert ca == cb
    assert np.abs(a - b).max() < 1e-3
