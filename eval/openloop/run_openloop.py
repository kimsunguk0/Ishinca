#!/usr/bin/env python3
"""Open-loop evaluation of a Pisco checkpoint (gold644-style protocol, see PROTOCOL.md).

  python eval/openloop/run_openloop.py --checkpoint ./Pisco-4B --data-root /data/clips \
      --manifest eval/openloop/manifests/gold644_clip_ids.txt --out runs/pisco4b_gold644

Per clip: t0 = 5.1 s, 4 cameras x 4 frames, 16-step history, no navigation input. Two independent calls:
rollout=1 (ADE/FDE) and rollout=6 (minADE6). Sampling T=0.6, top_p=0.98, bf16 autocast, torch seed reset to
42 before every call. Clips missing under --data-root are skipped and counted.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from metrics import clip_metrics
from pisco import build_model_inputs, load_pisco
from pisco.data import load_clip
from pisco.inputs import get_processor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="text file, one clip id per line")
    ap.add_argument("--data-root", help="per-clip zip root (data/README.md); omit for the official dataset interface")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dart-keep", type=float, default=None)
    ap.add_argument("--no-dart", action="store_true")
    ap.add_argument("--reference", type=Path, help="optional npz dir from gen_teacher_reference.py for the vs-Teacher metrics")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ids = [l.strip() for l in a.manifest.read_text().splitlines() if l.strip()]
    if a.limit:
        ids = ids[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paths").mkdir(exist_ok=True)
    dart = None if a.no_dart else (a.dart_keep if a.dart_keep is not None else "config")
    model = load_pisco(a.checkpoint, device=a.device, dart_keep=dart)
    processor = get_processor(model)
    rows, skipped = [], 0
    with (a.out / "results.jsonl").open("w", buffering=1) as f:
        for clip in ids:
            try:
                data = load_clip(clip, 5_100_000, a.data_root)
            except (FileNotFoundError, StopIteration):
                skipped += 1
                continue
            inputs = build_model_inputs(model, processor, data["image_frames"], data["camera_indices"],
                                        data["ego_history_xyz"], data["ego_history_rot"], a.device)
            out, secs = {}, {}
            for n in (1, 6):
                torch.manual_seed(a.seed)
                torch.cuda.manual_seed_all(a.seed)
                torch.cuda.synchronize()
                t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=n,
                        max_generation_length=256, return_extra=True)
                torch.cuda.synchronize()
                secs[n] = time.time() - t
                out[n] = xyz[0, 0].float().cpu().numpy()  # [n, 64, 3]
            gt = data["ego_future_xyz"][0, 0].numpy()
            row = {"clip_id": clip, **clip_metrics(out[1], out[6], gt), "sec_r1": secs[1], "sec_r6": secs[6]}
            ref = a.reference / f"{clip}.npz" if a.reference else None
            if ref is not None and ref.exists():
                tp = np.load(ref)["path0"]
                row.update({f"{k}_vs_teacher": v for k, v in clip_metrics(out[1], out[6], tp).items()})
            np.savez_compressed(a.out / "paths" / f"{clip}.npz", rollout1=out[1], rollout6=out[6])
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    keys = [k for k in rows[0] if k not in ("clip_id",)] if rows else []
    summary = {"n_clips": len(rows), "n_skipped_missing": skipped, "checkpoint": str(a.checkpoint),
               "dart_keep": model.dart_keep, **{k: float(np.mean([r[k] for r in rows])) for k in keys if k.startswith(("ade", "fde", "minade"))}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
