# Data: PhysicalAI-Autonomous-Vehicles

No data is shipped in this repository. Ishinca evaluates on clips of
[`nvidia/PhysicalAI-Autonomous-Vehicles`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
(gated; accept the NVIDIA Autonomous Vehicle Dataset License Agreement and log in with `hf auth login`).

## Option A: stream through the official interface

`scripts/infer.py --clip-id <uuid>` (without `--data-root`) calls `alpamayo1_5.load_physical_aiavdataset`, which uses
`physical_ai_av.PhysicalAIAVDatasetInterface` and downloads the needed chunks on demand. Simple, but each clip pulls
whole chunk zips (hundreds of MB) the first time.

## Option B: per-clip zips (offline, what the open-loop scripts use)

Cut the members of one clip out of the dataset's chunk zips and store them as one small zip per feature:

```
<data-root>/
  <clip_id>/
    egomotion.zip                       # <clip_id>.egomotion.parquet
    camera_cross_left_120fov.zip        # <clip_id>.camera_cross_left_120fov.{mp4,timestamps.parquet,blurred_boxes.parquet}
    camera_front_wide_120fov.zip
    camera_cross_right_120fov.zip
    camera_front_tele_30fov.zip
```

Source files in the dataset repository: `labels/egomotion/egomotion.chunk_XXXX.zip` and
`camera/<camera_name>/<camera_name>.chunk_XXXX.zip` (chunk index from `clip_index.parquet`). `eval/openloop/fetch_clips.py`
does the range-read + repack for a list of clip IDs (about 60 MB per clip).

Ishinca reads only: the 4 cameras above (camera indices 0, 1, 2, 6), 4 frames per camera at t0-0.3 s .. t0 (10 Hz),
16 ego poses ending at t0, and for evaluation 64 future poses (0.1 s steps). Default t0 = 5.1 s.

## Manifests

`eval/openloop/manifests/`:

* `gold644_clip_ids.txt`: the 644 clips of NVIDIA's public gold evaluation list, in original order.
* `gold644_teacher10b_reference.jsonl`: per-clip ADE / minADE6 (vs GT) of the Teacher 10B on gold644 (our run).
* `val4000_row_ids.json`: the 4000 internal val rows (row IDs, not clip IDs; mapping to clips is not included).
