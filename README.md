# Pisco

**Pisco: Distilling Alpamayo 1.5 (10B) into a 2B VLA.**

Pisco (a smaller neighbouring peak of Alpamayo) is a 2B Vision-Language-Action student of NVIDIA's
[Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) (the Teacher). Backbone: Cosmos-Reason2-2B
(Qwen3-VL-2B architecture), trained with image-relation distillation ("V-rel"), plus a 28-layer flow-matching action
expert (AE28). It takes 4 cameras x 4 frames and 16 ego-history poses and outputs a Chain-of-Causation (CoC) text and a
6.4 s trajectory (64 waypoints at 10 Hz). No navigation input. All distillation targets come from the Teacher; no
ground-truth trajectories are mixed into the targets.

| Model | Variant | Image tokens |
|---|---|---:|
| `suukim/Pisco-2B` | V-rel | 2,880 |
| `suukim/Pisco-2B-DART75` | + DART vision-token pruning, keep 75% (retrained) | 2,160 |
| `suukim/Pisco-2B-DART50` | + DART vision-token pruning, keep 50% (retrained) | 1,440 |

(`suukim` is a placeholder account; replace before publishing.)

This is a research release, not a production driving system.

## Install

```bash
pip install "git+https://github.com/NVlabs/alpamayo1.5.git" physical_ai_av   # upstream inference code (Apache-2.0)
pip install -e .                                                               # this repository
```

`transformers==4.57.1`, PyTorch >= 2.6 with CUDA. Tested on x86 (RTX 3090, torch 2.8 / CUDA 12.8) and aarch64 (RTX 5090,
torch 2.9.1); see `docs/validation_x86.md`. A Docker recipe is in `Dockerfile` (`docker build -t pisco:dev .`).
The model needs about 8 GiB of GPU memory in bf16.

## Quickstart

```bash
hf download suukim/Pisco-2B --local-dir ./Pisco-2B      # or pass the repo id directly

# a PhysicalAI-AV clip (gated dataset, see data/README.md)
python scripts/infer.py --checkpoint ./Pisco-2B --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 --seed 42
# DART variants read their keep fraction from config.json (key pisco_dart_keep); or force it:
python scripts/infer.py --checkpoint ./Pisco-2B-DART50 --dart-keep 0.5 --clip-id <uuid>
```

Python:

```python
import torch
from pisco import load_pisco, build_model_inputs
from pisco.data import load_clip
from pisco.inputs import get_processor

model = load_pisco("./Pisco-2B")                       # local dir or HF repo id; backbone weights are NOT needed
processor = get_processor(model)
data = load_clip("<clip-id>", t0_us=5_100_000)          # image_frames [4 cams,4,3,H,W], ego_history_xyz/rot
inputs = build_model_inputs(model, processor, data["image_frames"], data["camera_indices"],
                            data["ego_history_xyz"], data["ego_history_rot"])
torch.manual_seed(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    xyz, rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=1, max_generation_length=256, return_extra=True)
print(extra["cot"][0, 0, 0])       # CoC text
print(xyz[0, 0, 0, :, :2])         # 64 x 2, ego frame at t0, metres
```

`scripts/infer.py` also accepts `--inputs-npz` (decoded pixels saved with `--save-inputs`) so other machines replay the
identical input.

## Layout

```
pisco/                  model (Alpamayo1_5 subclass with the student tokenizer), loader, inputs, DART, clip loading
scripts/infer.py        frames + ego history -> CoC text + 6.4 s trajectory
eval/openloop/          protocol, metrics, run_openloop.py, Teacher reference generation, clip-ID manifests
eval/closedloop/        AlpaSim driver plugin + configs, topology, 150-scene list, scoring/re-aggregation
data/README.md          how to obtain PhysicalAI-AV and the expected directory structure (no data shipped)
docs/results.md         results; docs/validation_x86.md  x86 validation log
tests/                  DART unit tests, rollout-parity test (GPU)
```

How the code relates to upstream: Pisco does not copy NVIDIA's code. `PiscoModel` subclasses `alpamayo1_5.Alpamayo1_5`
and only swaps in the student's own tokenizer (16 special tokens, differs from the release tokenizer). Without DART the
upstream `sample_trajectories_from_data_with_vlm_rollout` runs unchanged; with DART an explicit rollout (prefill ->
CoC decode -> upstream flow-matching expert) is used. That rollout is bit-identical to upstream when nothing is pruned
(checked on one clip).

## DART

DART variants prune image tokens at the embedding level after the vision merger (per image `ceil(180 * keep)` of 180
merged tokens, pivots by L1 norm, duplicate removal by cosine similarity); kept tokens keep their original mRoPE
positions and every later position continues the full layout. The pruning is applied by the inference code, not stored
in the weights; running a DART checkpoint without pruning, or Pisco-2B with pruning, is not the evaluated configuration.
The eager PyTorch rollout here is not faster with DART (the speedups were measured in compiled/TensorRT runtimes).

## Results (summary)

Closed loop, AlpaSim, 150 scenes x 2 rollouts (Teacher 0.728, Pisco-2B 0.751, DART50 0.776, DART75 0.731; all paired
differences vs Teacher n.s.) and open loop: `docs/results.md`.

## License

* **Code in this repository**: Apache License 2.0 (`LICENSE`).
* **Model weights** (Hugging Face repos): NVIDIA Open Model License (backbone derived from Cosmos-Reason2-2B); Teacher /
  action-expert initialisation from Alpamayo-1.5-10B (OpenMDW-1.1); tokenizer files derive from Qwen3-VL (Apache-2.0).
  Where upstream terms differ, the more restrictive NVIDIA Open Model License applies to the weights.
* Dataset: NVIDIA Autonomous Vehicle Dataset License Agreement (PhysicalAI-AV); nothing is redistributed here.
* See `NOTICE`. Licensed by NVIDIA Corporation under the NVIDIA Open Model License. Built on NVIDIA Cosmos.
