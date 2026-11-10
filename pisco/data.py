"""Clip loading for PhysicalAI-Autonomous-Vehicles.

Two sources, both producing the dict of ``alpamayo1_5.load_physical_aiavdataset``:

* ``LocalClipStore``: per-clip feature zips on disk (see ``data/README.md``), no network.
* the official ``physical_ai_av`` dataset interface (Hugging Face; needs access to the gated dataset).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from physical_ai_av import egomotion, video

CAMERAS = {
    "CAMERA_CROSS_LEFT_120FOV": "camera_cross_left_120fov",
    "CAMERA_FRONT_WIDE_120FOV": "camera_front_wide_120fov",
    "CAMERA_CROSS_RIGHT_120FOV": "camera_cross_right_120fov",
    "CAMERA_FRONT_TELE_30FOV": "camera_front_tele_30fov",
}


class LocalClipStore:
    """Duck-typed stand-in for ``PhysicalAIAVDatasetInterface`` reading ``<root>/<clip_id>/<feature>.zip``."""

    features = SimpleNamespace(
        CAMERA=SimpleNamespace(**CAMERAS), LABELS=SimpleNamespace(EGOMOTION="egomotion")
    )

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _read(self, clip_id: str, feature: str, suffix: str) -> bytes:
        with zipfile.ZipFile(self.root / clip_id / f"{feature}.zip") as z:
            name = next(n for n in z.namelist() if n.endswith(suffix))
            return z.read(name)

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        if feature == "egomotion":
            df = pd.read_parquet(io.BytesIO(self._read(clip_id, feature, ".egomotion.parquet")))
            return egomotion.EgomotionState.from_egomotion_df(df).create_interpolator(
                df["timestamp"].to_numpy(copy=True)
            )
        ts = pd.read_parquet(io.BytesIO(self._read(clip_id, feature, ".timestamps.parquet")))
        return video.SeekVideoReader(
            video_data=io.BytesIO(self._read(clip_id, feature, ".mp4")),
            timestamps=ts["timestamp"].to_numpy(copy=True),
        )


def load_clip(clip_id: str, t0_us: int = 5_100_000, data_root: str | Path | None = None) -> dict[str, Any]:
    """Load one sample (4 cameras x 4 frames, 16-step history, 64-step future) at ``t0_us``."""
    if data_root is None:
        return load_physical_aiavdataset(clip_id, t0_us=t0_us)
    return load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=LocalClipStore(data_root), maybe_stream=False)


def save_inputs(path: str | Path, data: dict[str, Any]) -> None:
    """Dump decoded inputs (uint8 frames + ego history/future) so other hosts replay identical pixels."""
    keys = ["image_frames", "camera_indices", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot"]
    np.savez_compressed(path, **{k: data[k].numpy() for k in keys if k in data})


def load_inputs(path: str | Path) -> dict[str, Any]:
    z = np.load(path)
    return {k: torch.from_numpy(z[k]) for k in z.files}
