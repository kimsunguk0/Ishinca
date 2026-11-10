"""Open-loop metric definitions (XY, 64 future waypoints at 10 Hz = 6.4 s).

Only these metrics are reported for Pisco; never a 6-path mean or a min-of-5.

* ADE      : mean over the 64 waypoints of the XY distance between ONE independently generated path
             (a separate rollout=1 call) and the reference.
* FDE      : XY distance at the last waypoint (6.4 s) of that same path.
* minADE6  : minimum, over the 6 paths of a separate rollout=6 call, of the per-path ADE ("minADE6@6.4s").
* "vs GT"      : reference = recorded ego future.
* "vs Teacher" : reference = Teacher (Alpamayo 1.5 10B) AE path 0 generated from its own CoT with the same
                 inputs (see gen_teacher_reference.py). Ground truth is never mixed into either reference.
"""

from __future__ import annotations

import numpy as np


def path_ade(paths: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """paths [n, 64, >=2], ref [64, >=2] -> per-path ADE [n]."""
    d = np.linalg.norm(paths[..., :2].astype(np.float64) - ref[None, :, :2].astype(np.float64), axis=-1)
    return d.mean(-1)


def path_fde(paths: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(paths[..., :2].astype(np.float64) - ref[None, :, :2].astype(np.float64), axis=-1)
    return d[:, -1]


def clip_metrics(path1: np.ndarray, paths6: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """path1 [1, 64, 3] from the rollout=1 call, paths6 [6, 64, 3] from the rollout=6 call."""
    return {
        "ade": float(path_ade(path1, ref)[0]),
        "fde": float(path_fde(path1, ref)[0]),
        "minade6": float(path_ade(paths6, ref).min()),
    }
