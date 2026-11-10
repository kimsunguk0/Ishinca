#!/usr/bin/env python3
"""Pisco inference: 4 cameras x 4 frames + 16-step ego history -> Chain-of-Causation text + 6.4 s trajectory.

Examples:
  # a PhysicalAI-AV clip from the official dataset (needs HF access to the gated dataset)
  python scripts/infer.py --checkpoint suukim/Pisco-4B --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26

  # local per-clip zips (data/README.md), DART variant, deterministic seed
  python scripts/infer.py --checkpoint ./Pisco-4B-DART50 --data-root /data/clips --clip-id <uuid> --seed 42

  # replay exactly the same decoded pixels on another machine
  python scripts/infer.py --checkpoint ./Pisco-4B --inputs-npz inputs.npz --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pisco import build_model_inputs, load_pisco
from pisco.data import load_clip, load_inputs, save_inputs
from pisco.inputs import get_processor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="local directory or HF repo id")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip-id")
    src.add_argument("--inputs-npz", help="inputs saved with --save-inputs")
    ap.add_argument("--data-root", help="per-clip zip root; omit to use the official dataset interface")
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dart-keep", type=float, default=None, help="override DART keep fraction (default: from config)")
    ap.add_argument("--no-dart", action="store_true", help="disable DART even if the checkpoint config sets it")
    ap.add_argument("--pisco-rollout", action="store_true", help="use the explicit rollout even without DART")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("pisco_out.json"))
    ap.add_argument("--save-inputs", type=Path)
    a = ap.parse_args()

    dart = None if a.no_dart else (a.dart_keep if a.dart_keep is not None else "config")
    t = time.time()
    model = load_pisco(a.checkpoint, device=a.device, dart_keep=dart)
    processor = get_processor(model)
    t_load = time.time() - t

    data = load_inputs(a.inputs_npz) if a.inputs_npz else load_clip(a.clip_id, a.t0_us, a.data_root)
    if a.save_inputs:
        save_inputs(a.save_inputs, data)
    inputs = build_model_inputs(
        model, processor, data["image_frames"], data["camera_indices"],
        data["ego_history_xyz"], data["ego_history_rot"], a.device,
    )

    def run():
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.sample_trajectories_from_data_with_vlm_rollout(
                data=inputs, top_p=a.top_p, temperature=a.temperature, num_traj_samples=1,
                max_generation_length=a.max_new_tokens, noise_seed=a.seed,
                force_pisco_rollout=a.pisco_rollout, return_extra=True,
            )

    run()  # warm-up (kernel selection); same seed -> same output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    pred_xyz, pred_rot, extra = run()
    torch.cuda.synchronize()
    t_infer = time.time() - t

    xyz = pred_xyz[0, 0, 0].float().cpu().numpy()  # [64, 3] ego frame at t0, 0.1 s steps
    result = {
        "checkpoint": str(a.checkpoint), "clip_id": a.clip_id, "t0_us": a.t0_us, "seed": a.seed,
        "dart_keep": model.dart_keep, "coc": str(extra["cot"][0, 0, 0]),
        "pred_xy": xyz[:, :2].tolist(),
        "load_s": round(t_load, 2), "infer_s": round(t_infer, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "n_image_tokens_kept": None if model.dart_keep is None else f"keep={model.dart_keep}",
    }
    if "ego_future_xyz" in data:
        gt = data["ego_future_xyz"][0, 0, :, :2].numpy()
        d = np.linalg.norm(xyz[:, :2] - gt, axis=-1)
        result.update(ade_vs_gt=float(d.mean()), fde_vs_gt=float(d[-1]))
    a.out.write_text(json.dumps(result, indent=1))
    np.savez(a.out.with_suffix(".npz"), pred_xyz=xyz, pred_rot=pred_rot[0, 0, 0].float().cpu().numpy())
    print("CoC:", result["coc"])
    print({k: v for k, v in result.items() if k not in ("coc", "pred_xy")})


if __name__ == "__main__":
    main()
