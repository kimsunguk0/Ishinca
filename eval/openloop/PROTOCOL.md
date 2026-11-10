# Open-loop protocol

Metrics: ADE, FDE, minADE6@6.4 s, reported against the recorded ground truth ("vs GT") and against the Teacher
("vs Teacher"). Definitions are in `metrics.py`. A 6-path mean or a min-of-5 is never reported.

## Clip sets

| Set | File | Notes |
|---|---|---|
| gold644 | `manifests/gold644_clip_ids.txt` | 644 clips of NVIDIA's public gold evaluation list, original order, t0 = 5.1 s, no navigation. Primary external benchmark. |
| val4000 | `manifests/val4000_row_ids.json` | 4000 held-out internal rows (840 nav / 3160 no-nav), row IDs only. Used for development; not reproducible from this repo alone (row -> clip mapping is not included). `[TODO before publishing: export clip_id + t0 per row]` |

## Inputs and sampling (gold644)

* 4 cameras (indices 0, 1, 2, 6) x 4 frames at t0-0.3 s .. t0, 16-step ego history at 10 Hz, 64 future waypoints at 10 Hz.
* No navigation input, no Teacher CoT and no future target are given to generation.
* Prompt: `alpamayo1_5.helper.create_message`; images processed with the checkpoint's `backbone/` processor
  (min/max pixels 163840 / 196608).
* Two independent calls per clip: rollout = 1 (ADE, FDE) and rollout = 6 (minADE6, min over the six per-path ADEs).
  T = 0.6, top_p = 0.98, no top-k, `torch` seed reset to 42 before every call, bf16 autocast, batch 1, AE flow matching
  Euler 10 steps, max 256 generated tokens.
* DART variants run with their keep fraction (0.75 / 0.5); evaluating a DART checkpoint without pruning, or Pisco-2B with
  pruning, is not the evaluated configuration.

`run_openloop.py` implements this. The Teacher reference numbers in `manifests/gold644_teacher10b_reference.jsonl` were
produced with the same protocol by the original research code (FP16 + FlashAttention-2 there; bf16 + sdpa here, so small
numerical differences are expected).

## Same-x0 protocol (comparing two models' action-expert paths)

The flow-matching action expert starts from Gaussian noise x0. To compare two models (or two runtimes) without sampling
noise, both must integrate from the SAME x0: re-seed right before the sampler draws x0, not only at the start of the call
(the number of RNG draws consumed by CoC sampling differs between models and runtimes). `noise_seed=` in
`PiscoModel.sample_trajectories_from_data_with_vlm_rollout` does this. Pitfall: a global `torch.manual_seed` alone gives
different x0 per model whenever the generated CoC length differs.

"vs Teacher" additionally conditions on the CoC: Teacher AE path 0 from the Teacher's own CoT is the reference
(`gen_teacher_reference.py`), the Student path is generated from the Student's own CoT. Ground truth is never mixed into
either side.

## Streaming vs full window

Closed-loop and live stacks can reuse the KV of earlier frames (streaming). Measured on official644 x 17 samples:
streaming costs about +0.16 m ADE vs GT for the Student and +0.12 m for the 10B Teacher (same runtime), so it is not
Student-specific. This repository implements the full 16-frame window only.
