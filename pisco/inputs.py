"""Input preparation: 4 cameras x 4 frames + 16-step ego history -> model inputs.

Uses NVIDIA's ``alpamayo1_5.helper`` prompt (camera names, frame indices, history-trajectory placeholder).
The image processor comes from the checkpoint's ``backbone/`` folder (no Hub access needed).
"""

from __future__ import annotations

from typing import Any

import torch
from alpamayo1_5 import helper
from transformers import AutoProcessor


def get_processor(model) -> Any:
    cfg = model.config
    processor = AutoProcessor.from_pretrained(
        cfg.vlm_name_or_path, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels
    )
    processor.tokenizer = model.tokenizer
    return processor


def build_model_inputs(
    model,
    processor,
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Args follow ``alpamayo1_5.load_physical_aiavdataset``:

    image_frames ``[N_cam, 4, 3, H, W]`` uint8 (sorted by camera index), camera_indices ``[N_cam]``,
    ego_history_xyz ``[1, 1, 16, 3]`` and ego_history_rot ``[1, 1, 16, 3, 3]`` in the t0 ego frame.
    """
    messages = helper.create_message(frames=image_frames.flatten(0, 1), camera_indices=camera_indices)
    tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
        return_dict=True, return_tensors="pt",
    )
    return helper.to_device(
        {"tokenized_data": tokenized, "ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
        device,
    )
