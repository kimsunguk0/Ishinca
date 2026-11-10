"""CPU unit tests for the DART selection (no checkpoint needed): pytest tests/"""

import numpy as np
import torch

from pisco.dart import dart_keep, quota


def _toy(n_img=3, tok=180, d=32, n_text=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    parts, is_image = [torch.randn(n_text, d, generator=g)], [False] * n_text
    for _ in range(n_img):
        parts += [torch.randn(tok, d, generator=g), torch.randn(2, d, generator=g)]
        is_image += [True] * tok + [False] * 2
    return torch.cat(parts), np.array(is_image)


def test_quota_per_image():
    emb, is_img = _toy()
    for keep in (1.0, 0.75, 0.5):
        flags = dart_keep(emb, is_img, keep)
        assert flags.shape == (is_img.sum(),)
        assert flags.reshape(3, 180).sum(1).tolist() == [quota(180, keep)] * 3


def test_keep_all_is_identity():
    emb, is_img = _toy()
    assert dart_keep(emb, is_img, 1.0).all()


def test_deterministic():
    emb, is_img = _toy()
    assert (dart_keep(emb, is_img, 0.5) == dart_keep(emb, is_img, 0.5)).all()
