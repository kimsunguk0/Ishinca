"""Ishinca model: Alpamayo 1.5 architecture (Qwen3-VL-2B backbone + AE28 action expert) with the student tokenizer.

Everything except the tokenizer and the optional DART rollout is inherited from NVIDIA's
``alpamayo1_5`` package (VLM/expert construction, weight loading, trajectory tokens, flow matching,
action space). The student ships its own tokenizer (16 special tokens, e.g. ``<|meta_action_*|>``) that
differs from the release ``alpamayo1_5`` tokenizer, so it is loaded verbatim instead of rebuilt.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import einops
import numpy as np
import torch
from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.base_model import TRAJ_TOKEN
from alpamayo1_5.models.token_utils import extract_text_tokens
from transformers import AutoProcessor, DynamicCache

from ishinca.dart import dart_keep as select_dart_tokens

logger = logging.getLogger(__name__)


def load_student_tokenizer(vlm_name_or_path: str, min_pixels: int | None, max_pixels: int | None) -> Any:
    """Load the tokenizer bundled with the checkpoint as-is (trajectory/special tokens baked in)."""
    kwargs = {k: v for k, v in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if v is not None}
    tokenizer = AutoProcessor.from_pretrained(vlm_name_or_path, **kwargs).tokenizer
    tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
    tokenizer.traj_token_ids = {k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()}
    return tokenizer


class IshincaConfig(Alpamayo1_5Config):
    """Alpamayo 1.5 config that reads vocabulary metadata from the student tokenizer.

    ``ishinca_dart_keep`` (optional, in config.json) is the DART keep fraction of the checkpoint
    (0.5 / 0.75 for the DART variants, absent for Ishinca-4B).
    """

    def __init__(self, ishinca_dart_keep: float | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ishinca_dart_keep = ishinca_dart_keep

    def _initialize_vlm_config(self) -> None:
        if self.vlm_name_or_path is None:
            return
        tokenizer = load_student_tokenizer(self.vlm_name_or_path, self.min_pixels, self.max_pixels)
        self.vocab_size = len(tokenizer)
        self.traj_token_start_idx = tokenizer.traj_token_start_idx
        self.traj_token_ids = tokenizer.traj_token_ids


class _LinearPatchProj(torch.nn.Module):
    """Conv3d(kernel == stride) patch projection as one matmul (mathematically identical).

    The Qwen3-VL vision patch embedding is a Conv3d whose kernel equals its stride, i.e. a linear map of each flattened
    patch. cuDNN's Conv3d is extremely slow on some GPUs/builds (measured: ~190 s per inference on an RTX 5090 with
    torch 2.9.1 / aarch64, vs ~1 s with a matmul); a plain matmul avoids it.
    """

    def __init__(self, conv: torch.nn.Conv3d) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(conv.weight.detach().flatten(1).contiguous(), requires_grad=False)
        self.bias = torch.nn.Parameter(conv.bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [N, C, T, P, P]
        return torch.nn.functional.linear(x.flatten(1), self.weight, self.bias)


def use_linear_patch_embed(model: "IshincaModel") -> None:
    embed = model.vlm.model.visual.patch_embed
    if isinstance(embed.proj, torch.nn.Conv3d):
        embed.proj = _LinearPatchProj(embed.proj)


def _top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Temperature + nucleus sampling on ``[1, V]`` logits (same rule as HF ``TopPLogitsWarper``)."""
    logits = logits.float() / temperature
    sorted_logits, sorted_idx = torch.sort(logits, descending=False)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    remove = cum <= (1 - top_p)
    remove[..., -1:] = False
    logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1)


class IshincaModel(Alpamayo1_5):
    config_class = IshincaConfig

    dart_keep: float | None = None  # set by load_ishinca / set_dart

    def _build_tokenizer(self, config: IshincaConfig) -> Any:
        return load_student_tokenizer(config.vlm_name_or_path, config.min_pixels, config.max_pixels)

    def set_dart(self, keep_frac: float | None) -> None:
        """Enable DART image-token reduction (keep fraction per image) or disable it with ``None``."""
        self.dart_keep = None if keep_frac is None else float(keep_frac)

    # ------------------------------------------------------------------ sampling

    @torch.inference_mode()
    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
        diffusion_kwargs: dict[str, Any] | None = None,
        noise_seed: int | None = None,
        force_ishinca_rollout: bool = False,
        return_extra: bool = False,
        **kwargs: Any,
    ):
        """Same contract as the upstream method (returns ``pred_xyz, pred_rot[, extra]``).

        Without DART this calls the upstream implementation (HF ``generate``), unless
        ``force_ishinca_rollout`` is set. With DART, or when forced, the explicit rollout below is used;
        it draws ``num_traj_samples`` independent samples sequentially.
        ``noise_seed`` re-seeds the RNG right before the flow-matching sampler draws x0 (same-x0 protocol).
        """
        if self.dart_keep is None and not force_ishinca_rollout:
            if noise_seed is not None:
                self._seed_diffusion_noise(noise_seed)
            try:
                return super().sample_trajectories_from_data_with_vlm_rollout(
                    data=data, top_p=top_p, temperature=temperature, num_traj_samples=num_traj_samples,
                    max_generation_length=max_generation_length, diffusion_kwargs=diffusion_kwargs,
                    return_extra=return_extra, **kwargs,
                )
            finally:
                self._restore_diffusion_sample()
        outs = [
            self.rollout(data, top_p, temperature, max_generation_length, diffusion_kwargs,
                         None if noise_seed is None else noise_seed + i)
            for i in range(num_traj_samples)
        ]
        pred_xyz = torch.cat([o[0] for o in outs], dim=2)
        pred_rot = torch.cat([o[1] for o in outs], dim=2)
        if not return_extra:
            return pred_xyz, pred_rot
        extra = {k: np.concatenate([o[2][k] for o in outs], axis=2) for k in outs[0][2]}
        return pred_xyz, pred_rot, extra

    def _seed_diffusion_noise(self, seed: int) -> None:
        if getattr(self, "_orig_diffusion_sample", None) is None:
            self._orig_diffusion_sample = self.diffusion.sample

            def sample(*a: Any, **k: Any):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                return self._orig_diffusion_sample(*a, **k)

            self.diffusion.sample = sample

    def _restore_diffusion_sample(self) -> None:
        if getattr(self, "_orig_diffusion_sample", None) is not None:
            self.diffusion.sample = self._orig_diffusion_sample
            self._orig_diffusion_sample = None

    def rollout(
        self,
        data: dict[str, Any],
        top_p: float,
        temperature: float,
        max_new_tokens: int,
        diffusion_kwargs: dict[str, Any] | None,
        noise_seed: int | None,
    ):
        """Prefill (optionally DART-reduced) -> CoC decode until ``<|traj_future_start|>`` -> action expert."""
        data = copy.deepcopy(data)
        tok = data["tokenized_data"]
        hist = {"ego_history_xyz": data["ego_history_xyz"], "ego_history_rot": data["ego_history_rot"]}
        input_ids = self.fuse_traj_tokens(tok["input_ids"], hist)
        assert input_ids.shape[0] == 1, "batch size 1 only"
        device = input_ids.device
        core, lm_head = self.vlm.model, self.vlm.lm_head
        text_model = core.language_model

        # --- embeddings: text + merged image features (+ DeepStack features)
        embeds = text_model.embed_tokens(input_ids)
        image_embeds, deepstack = core.get_image_features(tok["pixel_values"], tok["image_grid_thw"])
        image_embeds = torch.cat(image_embeds).to(embeds.dtype)
        image_mask = input_ids == self.vlm.config.image_token_id
        embeds = embeds.masked_scatter(image_mask[..., None].expand_as(embeds), image_embeds)
        position_ids, _ = core.get_rope_index(input_ids, tok["image_grid_thw"], None, tok.get("attention_mask"))
        next_pos = int(position_ids.max()) + 1  # first position of the first generated token

        # --- DART: drop image tokens; kept tokens keep their full-layout mRoPE positions
        if self.dart_keep is not None:
            is_image = image_mask[0].cpu().numpy()
            keep = select_dart_tokens(embeds[0], is_image, self.dart_keep)
            drop = np.zeros(is_image.shape[0], bool)
            drop[np.where(is_image)[0][~keep]] = True
            kept = torch.from_numpy(np.where(~drop)[0]).to(device)
            embeds, position_ids, image_mask = embeds[:, kept], position_ids[:, :, kept], image_mask[:, kept]
            deepstack = [d[torch.from_numpy(keep).to(device)] for d in deepstack]
        prompt_len = embeds.shape[1]

        # --- prefill
        cache = DynamicCache(config=text_model.config)
        out = text_model(
            inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(prompt_len, device=device),
            visual_pos_masks=image_mask, deepstack_visual_embeds=deepstack,
        )
        logits = lm_head(out.last_hidden_state[:, -1:])[:, 0]

        # --- CoC decode: no trajectory tokens, stop once <|traj_future_start|> has its KV written
        stop_id = self.tokenizer.convert_tokens_to_ids(TRAJ_TOKEN["future_start"])
        lo = self.config.traj_token_start_idx
        generated: list[torch.Tensor] = []
        for i in range(max_new_tokens):
            logits[:, lo : lo + self.config.traj_vocab_size] = float("-inf")
            token = _top_p_sample(logits, temperature, top_p)
            generated.append(token)
            out = text_model(
                input_ids=token, position_ids=torch.full((3, 1, 1), next_pos + i, device=device),
                past_key_values=cache, use_cache=True,
                cache_position=torch.tensor([prompt_len + i], device=device),
            )
            logits = lm_head(out.last_hidden_state[:, -1:])[:, 0]
            if int(token) == stop_id:
                break
        else:  # budget exhausted: force the trajectory-start token so the expert reads its KV
            token = torch.full((1, 1), stop_id, device=device)
            generated.append(token)
            text_model(input_ids=token, position_ids=torch.full((3, 1, 1), next_pos + max_new_tokens, device=device),
                       past_key_values=cache, use_cache=True,
                       cache_position=torch.tensor([prompt_len + max_new_tokens], device=device))
        kv_len = cache.get_seq_length()  # prompt + generated tokens, ends with <|traj_future_start|>
        sequences = torch.cat([input_ids, *generated], dim=1)

        # --- action expert (upstream step function / flow matching)
        n_diff = self.action_space.get_action_space_dims()[0]
        exp_pos, exp_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=torch.tensor([kv_len], device=device),
            rope_deltas=torch.tensor([[next_pos + len(generated) - kv_len]], device=device),
            kv_cache_seq_len=kv_len, n_diffusion_tokens=n_diff, b_star=1, device=device,
        )
        fwd = {"is_causal": False} if self.config.expert_non_causal_attention else {}

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            emb = self.action_in_proj(x, t)
            if emb.dim() == 2:
                emb = emb.view(x.shape[0], n_diff, -1)
            hidden = self.expert(inputs_embeds=emb, position_ids=exp_pos, past_key_values=cache,
                                 attention_mask=exp_mask, use_cache=True, **fwd).last_hidden_state
            cache.crop(kv_len)
            return self.action_out_proj(hidden[:, -n_diff:]).view(-1, *self.action_space.get_action_space_dims())

        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            torch.cuda.manual_seed_all(noise_seed)
        action = self.diffusion.sample(batch_size=1, step_fn=step_fn, device=device, return_all_steps=False,
                                       **(diffusion_kwargs or {}))
        hx = einops.repeat(data["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=1)
        hr = einops.repeat(data["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=1)
        xyz, rot = self.action_space.action_to_traj(action, hx, hr)
        xyz = einops.rearrange(xyz, "(b ns nj) ... -> b ns nj ...", ns=1, nj=1)
        rot = einops.rearrange(rot, "(b ns nj) ... -> b ns nj ...", ns=1, nj=1)
        extra = extract_text_tokens(self.tokenizer, sequences)
        extra = {k: np.array(v).reshape([1, 1, 1]) for k, v in extra.items()}
        return xyz, rot, extra
