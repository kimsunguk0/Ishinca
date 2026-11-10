#!/usr/bin/env python3
"""Download selected PhysicalAI-AV clips into the per-clip zip layout of data/README.md (needs HF access).

  python eval/openloop/fetch_clips.py --clips eval/openloop/manifests/gold644_clip_ids.txt --out /data/clips [--limit N]

Uses the dataset's clip_index.parquet to find the chunk of each clip, then range-reads only that clip's zip members
(``HfFileSystem``) and repacks them. [UNVERIFIED: clip_index.parquet columns checked, but the download itself was
not executed in this repository's validation.]
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
FEATURES = {
    "egomotion": ("labels/egomotion", ["egomotion.parquet"]),
    **{c: (f"camera/{c}", [f"{c}.mp4", f"{c}.timestamps.parquet", f"{c}.blurred_boxes.parquet"])
       for c in ("camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_front_tele_30fov")},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ids = [l.strip() for l in a.clips.read_text().splitlines() if l.strip()]
    ids = ids[: a.limit] if a.limit else ids
    index = pd.read_parquet(hf_hub_download(REPO, "clip_index.parquet", repo_type="dataset"))
    fs = HfFileSystem()
    for clip in ids:
        chunk = int(index.loc[clip, "chunk"])  # clip_index.parquet: index clip_id, columns clip_is_valid / chunk / split
        for feat, (folder, members) in FEATURES.items():
            dest = a.out / clip / f"{feat}.zip"
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = f"datasets/{REPO}/{folder}/{Path(folder).name}.chunk_{chunk:04d}.zip"
            with fs.open(src, "rb", block_size=8 << 20) as s, zipfile.ZipFile(s) as remote, \
                    zipfile.ZipFile(dest.with_suffix(".part"), "w", zipfile.ZIP_STORED, allowZip64=True) as local:
                for m in members:
                    with remote.open(f"{clip}.{m}") as r, local.open(f"{clip}.{m}", "w", force_zip64=True) as w:
                        shutil.copyfileobj(r, w, 4 << 20)
            dest.with_suffix(".part").replace(dest)


if __name__ == "__main__":
    main()
