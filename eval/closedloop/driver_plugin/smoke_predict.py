#!/usr/bin/env python3
"""Smoke test of the plugin without the simulator: feed one saved sample through AlpaSim's driver `predict()`.

  python smoke_predict.py --checkpoint ./Pisco-2B --inputs-npz inputs.npz [--reference pisco_out.npz]

inputs.npz comes from `scripts/infer.py --save-inputs`. With --reference (the .npz written by infer.py for the same
checkpoint and seed 42) the printed difference should be ~0: the driver path and infer.py then run the same model on
the same pixels and the same flow-matching x0.
"""
import argparse
import sys
import types
from pathlib import Path

import numpy as np
import scipy.spatial.transform as spt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from alpasim_driver.models.base import CameraFrame, DriveCommand, PredictionInput  # noqa: E402
from alpasim_grpc.v0.common_pb2 import PoseAtTime  # noqa: E402
from pisco_alpasim.model import PiscoDriverModel  # noqa: E402

NAMES = {0: "camera_cross_left_120fov", 1: "camera_front_wide_120fov", 2: "camera_cross_right_120fov", 6: "camera_front_tele_30fov"}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--inputs-npz", required=True)
ap.add_argument("--reference")
a = ap.parse_args()

z = np.load(a.inputs_npz)
frames, cams = z["image_frames"], z["camera_indices"].tolist()
t0 = 5_100_000
imgs = {NAMES[c]: [CameraFrame(t0 - (3 - i) * 100_000, frames[k, i].transpose(1, 2, 0).copy()) for i in range(4)]
        for k, c in enumerate(cams)}
xyz, quat = z["ego_history_xyz"][0, 0], spt.Rotation.from_matrix(z["ego_history_rot"][0, 0]).as_quat()
poses = []
for i in range(16):
    p = PoseAtTime()
    p.timestamp_us = t0 - (15 - i) * 100_000
    p.pose.vec.x, p.pose.vec.y, p.pose.vec.z = map(float, xyz[i])
    p.pose.quat.x, p.pose.quat.y, p.pose.quat.z, p.pose.quat.w = map(float, quat[i])
    poses.append(p)
cfg = types.SimpleNamespace(checkpoint_path=a.checkpoint, compiled_runtime=False, vlm_temperature=None, vlm_top_p=None)
model = PiscoDriverModel.from_config(cfg, torch.device("cuda"), list(NAMES.values()), 4, 10)
torch.manual_seed(42)
model._model._seed_diffusion_noise(42)  # same x0 as infer.py --seed 42
out = model.predict(PredictionInput(camera_images=imgs, command=DriveCommand.UNKNOWN, speed=0.0, acceleration=0.0,
                                    ego_pose_history=poses, inference_seed=42))
print("trajectory", out.trajectory_xy.shape, "CoC:", out.reasoning_text)
if a.reference:
    ref = np.load(a.reference)["pred_xyz"][:, :2]
    print("mean |dxy| vs reference: %.4f" % np.linalg.norm(out.trajectory_xy - ref, axis=-1).mean())
