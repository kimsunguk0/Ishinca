#!/usr/bin/env python3
"""Generate the Teacher reference paths (Alpamayo 1.5 10B, upstream code, own CoT, AE path 0).

  python eval/openloop/gen_teacher_reference.py --manifest manifests/gold644_clip_ids.txt \
      --data-root /data/clips --out refs/teacher10b

Writes <out>/<clip_id>.npz with ``path0`` [64, 3] (rollout=1) and ``paths6`` [6, 64, 3] (rollout=6), same protocol
as run_openloop.py (T=0.6, top_p=0.98, seed 42 per call, bf16 autocast). Needs ~24+ GB of GPU memory for the 10B
weights. [UNVERIFIED: written against the upstream API; not executed in this repository's validation.]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from ishinca.data import load_clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = Alpamayo1_5.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    for clip in [l.strip() for l in a.manifest.read_text().splitlines() if l.strip()]:
        if (a.out / f"{clip}.npz").exists():
            continue
        try:
            data = load_clip(clip, 5_100_000, a.data_root)
        except (FileNotFoundError, StopIteration):
            continue
        msgs = helper.create_message(data["image_frames"].flatten(0, 1), data["camera_indices"])
        tok = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                            continue_final_message=True, return_dict=True, return_tensors="pt")
        inp = helper.to_device({"tokenized_data": tok, "ego_history_xyz": data["ego_history_xyz"],
                                "ego_history_rot": data["ego_history_rot"]}, "cuda")
        res = {}
        for n in (1, 6):
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=inp, top_p=0.98, temperature=0.6, num_traj_samples=n, max_generation_length=256)
            res[n] = xyz[0, 0].float().cpu().numpy()
        np.savez_compressed(a.out / f"{clip}.npz", path0=res[1][0], paths6=res[6])


if __name__ == "__main__":
    main()
