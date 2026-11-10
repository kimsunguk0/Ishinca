"""Checkpoint loading: local directory or Hugging Face repo id (flat layout, no subfolders)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ishinca.modeling import IshincaConfig, IshincaModel, use_linear_patch_embed

# A Ishinca checkpoint directory contains: config.json (vlm_name_or_path = "./backbone"), model.safetensors
# (full model: VLM + AE28), and backbone/ (tokenizer + processor + backbone config.json; NO backbone weights,
# the VLM weights are inside model.safetensors).


def resolve_checkpoint(path_or_id: str, revision: str | None = None) -> Path:
    p = Path(path_or_id).expanduser()
    if p.is_dir():
        return p.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_id, revision=revision))


def load_ishinca(
    path_or_id: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    dart_keep: float | None | str = "config",
    attn_implementation: str = "sdpa",
    revision: str | None = None,
    linear_patch_embed: bool = True,
) -> IshincaModel:
    """Load a Ishinca checkpoint.

    Args:
        path_or_id: local directory or HF repo id (e.g. ``suukim/Ishinca-4B``).
        dart_keep: DART keep fraction. ``"config"`` (default) uses ``ishinca_dart_keep`` from config.json
            (set for the DART variants), ``None`` disables DART.
        linear_patch_embed: run the vision patch-embedding Conv3d as an equivalent matmul (same math; avoids a very
            slow cuDNN Conv3d path seen on RTX 5090 / aarch64). Set False for the stock Conv3d.
    """
    root = resolve_checkpoint(path_or_id, revision)
    cfg_dict = json.loads((root / "config.json").read_text())
    cfg_dict["vlm_name_or_path"] = str((root / cfg_dict.get("vlm_name_or_path", "./backbone")).resolve())
    cfg_dict["attn_implementation"] = attn_implementation
    config = IshincaConfig(**cfg_dict)
    model = IshincaModel.from_pretrained(root, config=config, dtype=dtype, attn_implementation=attn_implementation)
    model = model.to(device).eval().requires_grad_(False)
    if linear_patch_embed:
        use_linear_patch_embed(model)
    model.set_dart(config.ishinca_dart_keep if dart_keep == "config" else dart_keep)
    return model
