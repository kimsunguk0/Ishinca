# x86 validation log (2026-11-09)

Goal: show that a Pisco checkpoint in the Hugging Face layout runs from this repository on x86 in Docker, and how it
compares with the aarch64 path. All numbers below were measured in this session unless marked otherwise.

## Setup

| | x86 | aarch64 reference |
|---|---|---|
| GPU | RTX 3090 24 GB (sm_86) | RTX 5090 32 GB (sm_120) |
| Runtime | Docker image `pisco:dev` = `alpamayo15-flash` base (torch 2.8.0 / CUDA 12.8, transformers 4.57.1, alpamayo1_5 0.1.0, physical_ai_av 0.2.0) + this repo (`pip install --no-deps -e .`) | existing driver venv in a container (torch 2.9.1 / CUDA 12.8, transformers 4.57.1), repo on `PYTHONPATH` |
| Weights | Pisco-4B, DART50, DART75 copied from the HF staging folder (`tar` stream, symlinks resolved); sha256 of all three `model.safetensors` equals the release list | same files (symlinks inside the container) |
| Inputs | real PhysicalAI-AV clips, per-clip zips (8 gold644 clips available locally) | decoded pixels of one clip dumped by the x86 run (`--save-inputs` / `--inputs-npz`), so both hosts see identical tensors |

## Loading without `backbone/model.safetensors`

Works. The checkpoint folder is `config.json` + `model.safetensors` + `backbone/` (tokenizer, processor, backbone
`config.json` only). `PiscoModel` builds the VLM from `backbone/config.json` (random init) and `model.safetensors`
holds all VLM and action-expert weights (`PACKAGE.json`: 626 VLM + 321 expert tensors, no missing / unexpected keys).
The loader resolves `vlm_name_or_path: ./backbone` relative to the checkpoint folder, so `scripts/localize_config.py`
is not needed. HF Hub was disabled (`HF_HUB_OFFLINE=1`) during all runs: nothing is fetched at load time.

## Results

Single clip (10c18013..., t0 = 5.1 s), seed 42, bf16, full CoC + 10-step flow matching:

| check | result |
|---|---|
| x86 `scripts/infer.py` Pisco-4B | CoC "Nudge to the right to clear the construction barrels blocking the left side of our lane"; ADE 1.18 m, FDE 5.93 m vs GT |
| latency (RTX 3090, warm, one call, 4 cams x 4 frames) | Pisco-4B 1.1-1.2 s; DART50 1.0 s; DART75 1.1 s (eager PyTorch, no CUDA graphs) |
| peak VRAM | 8.0 GiB (Pisco-4B), 7.9 GiB (DART) |
| model load | 4-16 s (page cache dependent) |
| explicit Pisco rollout vs upstream `generate` path (x86, same x0) | CoC equal, path difference 0.0 m (T = 0.01 and T = 0.6) |
| DART keep = 1.0 vs no DART | identical (0.0 m): the pruning path is a no-op when nothing is dropped |
| DART keep 0.75 / 0.5 vs full tokens (same checkpoint, same x0) | path shift 0.30 m / 0.90 m on this clip (CoC unchanged); this is expected: the retrained DART checkpoints are meant to run pruned |
| x86 vs aarch64, Pisco-4B, T = 0.6, seed 42, identical pixels | CoC identical; path mean difference 0.9 cm, max 4.1 cm |
| x86 vs aarch64, DART50, T = 0.01 (near-greedy) | CoC identical; mean 2.1 cm, max 8.2 cm |
| x86 vs aarch64, DART50, T = 0.6 | CoC differs ("Lane change to the left ..." vs "Nudge to the right ..."), path mean 0.72 m, max 2.4 m |
| stock Conv3d vs matmul patch embedding (same host) | mean 1.9 cm (x86), 1.4 cm (aarch64): bf16 rounding only |
| AlpaSim driver `predict()` (plugin, aarch64 driver env) vs `infer.py`, same x0 | 0.000 m |

Tolerance, stated honestly: with the same seed, same pixels and same x0, x86 and aarch64 agree to about 1-2 cm over
64 waypoints when the sampled CoC is the same, which is bf16-level noise. The CoC is sampled (T = 0.6, top-p 0.98); a
1-bf16-ulp logit difference can flip a near-tied token, and then the CoC and the path legitimately differ (seen once in
DART50, where the pruned model's token distribution is flatter). Hence cross-host bit equality is not expected and the
test uses near-greedy sampling (T = 0.01) for DART. Only one clip was used for the cross-host comparison.

Cross-check against the research evaluation code (same 8 gold644 clips, Pisco-4B, bf16, seed 42 per call, rollout=1):
per-clip ADE of this repository vs the original per-clip result of the research benchmark run (V-rel bundle of
2026-10-07, FlashAttention-2, stock Conv3d): differences <= 0.03 m on every clip (mean 2.420 vs 2.421 m). minADE6 is a
minimum over six random draws whose RNG order differs between the two implementations (batched `generate` there), so
it agrees only statistically (mean 1.069 vs 1.229 m on 8 clips, per-clip differences up to 1 m); do not compare it
clip by clip. This also supports (without proving) that the staged `Pisco-4B` weights equal that bundle.

Open-loop smoke over the 8 gold644 clips on x86 (n = 8, not a benchmark; `docs/results.md`): ADE / FDE / minADE6 =
2.420 / 6.352 / 1.069 (Pisco-4B), 2.399 / 7.572 / 1.069 (DART50), 2.245 / 6.679 / 1.016 (DART75) m. rollout=1 latency median
1.1 s (DART50 1.0 s); rollout=6 (six sequential samples for DART, batched generation for Pisco-4B) 4.0 s / 6.2 s / 6.3 s.

## Problems found and fixed

1. **cuDNN Conv3d patch embedding is catastrophically slow on RTX 5090 / aarch64 / torch 2.9.1**: one inference took
   184-192 s (GPU 8% busy, one CPU core at 100%; stack dump pointed to `Qwen3VLVisionPatchEmbed.proj` Conv3d). The patch
   projection has kernel == stride, so it is exactly a matmul; `pisco.modeling.use_linear_patch_embed` (default on in
   `load_pisco`) replaces it: 192 s -> 2.3 s on the 5090. On x86 it is neutral (1.2 s either way). This is the same
   substitution the compiled AlpaSim runtime applies. Set `linear_patch_embed=False` for the stock op.
2. The draft HF README and `scripts/localize_config.py` assume an absolute `vlm_name_or_path`; the new loader resolves
   the relative path itself.
3. The DART variants' `config.json` does not carry the keep fraction (identical to Pisco-4B's file). The loader accepts
   `dart_keep=` / `--dart-keep` and reads `pisco_dart_keep` from config.json if present (see TODO).
4. Importing `pisco.dart` pulled in torch/transformers through `pisco/__init__`; made lazy so unit tests run on CPU.

## What was not validated

* Full closed-loop simulation with the Pisco plugin: a 24 GB card cannot hold the NRE renderer (peaks ~18 GB in our
  telemetry) and the driver, and only one x86 GPU was available, so no AlpaSim rollout was run on x86. The plugin was
  checked through the driver's `predict()` on a saved sample (aarch64 driver env), not through the simulator, and not via
  the `alpasim.models` entry point.
* The 10B Teacher reference script (`gen_teacher_reference.py`) and `fetch_clips.py` were not executed (24+ GB model;
  gated dataset download).
* Streaming-KV runtime, compiled/CUDA-graph runtime and TensorRT paths are not part of this repository.
* Cross-host numerical comparison used one clip; the 8-clip x86 run has no aarch64 counterpart.
