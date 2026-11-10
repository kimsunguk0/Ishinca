"""DART vision-token selection used by the Ishinca-4B-DART variants.

Port of the selection the DART Students were trained and evaluated with (NVIDIA DartPruner
semantics): per image, keep a fixed quota (``ceil(keep * n_tokens)``) of the merged image tokens.
Pivots are each image's top-4 L1-norm embeddings plus the prompt's top-4 L1-norm text embeddings;
the remaining slots are filled by lowest cosine similarity to the pivots (duplicate-token removal).
All norms and dot products are computed in FP32.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _top(pos: np.ndarray, score: np.ndarray, k: int) -> np.ndarray:
    """Top-k of ``pos`` by descending score, ties broken by lower position."""
    return pos[np.lexsort((pos, -score[pos]))[: max(k, 0)]]


def quota(n_tokens: int, keep_frac: float) -> int:
    return max(1, min(n_tokens, math.ceil(n_tokens * keep_frac)))


@torch.no_grad()
def dart_keep(
    embeds: torch.Tensor,
    is_image: np.ndarray,
    keep_frac: float,
    pivot_image: int = 4,
    pivot_text: int = 4,
) -> np.ndarray:
    """Keep flag per image token.

    Args:
        embeds: ``[L, D]`` prompt input embeddings (image rows = merger output).
        is_image: ``[L]`` bool, True at image-token positions.
        keep_frac: fraction of each image's tokens to keep (0.75 / 0.5).

    Returns:
        ``[n_image_tokens]`` bool array in prompt order.
    """
    with torch.autocast(embeds.device.type, enabled=False):
        e = embeds.float()
        l1 = e.abs().sum(-1).cpu().numpy()
        l2 = e.norm(dim=-1).cpu().numpy()
        img = np.where(is_image)[0]
        txt = np.where(~is_image)[0]
        txt_piv = _top(txt, l1, pivot_text)
        spans = np.split(img, np.where(np.diff(img) > 1)[0] + 1)
        span_piv = [_top(s, l1, min(pivot_image, quota(len(s), keep_frac))) for s in spans]
        piv = np.concatenate(span_piv + [txt_piv])
        dots = (e[torch.from_numpy(piv).to(e.device)] @ e.T).cpu().numpy()
    row = {int(p): j for j, p in enumerate(piv)}
    kept: list[int] = []
    for s, sp in zip(spans, span_piv):
        target = quota(len(s), keep_frac)
        mine = list(sp)
        cand = np.setdiff1d(s, sp)
        order = list(sp) + list(txt_piv)
        sim = np.zeros(len(l1))
        for k, p in enumerate(order):
            rem = target - len(mine)
            if rem <= 0 or not len(cand):
                break
            per = -(-rem // (len(order) - k))
            sim[cand] = -(dots[row[int(p)], cand] / np.maximum(l2[p] * l2[cand], 1e-8))
            chosen = _top(cand, sim, per)
            mine += chosen.tolist()
            cand = np.setdiff1d(cand, chosen)
        assert len(mine) == target, (len(mine), target)
        kept += mine
    keep = np.zeros(len(img), bool)
    keep[np.searchsorted(img, np.asarray(kept))] = True
    return keep
