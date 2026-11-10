"""AlpaSim driver adapter for Ishinca.

AlpaSim resolves ``model.model_type`` through the ``alpasim.models`` entry-point group; this package registers
``ishinca``. Frame selection, ego-history construction, prompt building and the call into
``sample_trajectories_from_data_with_vlm_rollout`` are inherited from AlpaSim's ``Alpamayo15Model``; only the
checkpoint loading is replaced by :func:`ishinca.load_ishinca`.

DART: the keep fraction comes from ``ishinca_dart_keep`` in the checkpoint's config.json, or from the environment
variable ``ISHINCA_DART_KEEP`` (``none`` disables DART). The AlpaSim ``ModelConfig`` schema is not modified.
"""

from __future__ import annotations

import logging
import os

import torch
from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
from ishinca import load_ishinca
from ishinca.inputs import get_processor

logger = logging.getLogger(__name__)


class IshincaDriverModel(Alpamayo15Model):
    """Ishinca-4B / Ishinca-4B-DART50 / Ishinca-4B-DART75 behind the AlpaSim driver interface (eager PyTorch rollout)."""

    @classmethod
    def from_config(cls, model_cfg, device, camera_ids, context_length, output_frequency_hz):
        if getattr(model_cfg, "compiled_runtime", False):
            raise ValueError("model_type 'ishinca' uses the eager rollout; set compiled_runtime: false")
        model = cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
        )
        if getattr(model_cfg, "vlm_temperature", None) is not None:
            model._temperature = float(model_cfg.vlm_temperature)
        if getattr(model_cfg, "vlm_top_p", None) is not None:
            model._top_p = float(model_cfg.vlm_top_p)
        return model

    def __init__(self, checkpoint_path, device, camera_ids, context_length=Alpamayo15Model.DEFAULT_CONTEXT_LENGTH,
                 num_traj_samples=1, top_p=0.98, temperature=0.6, **_):
        env = os.environ.get("ISHINCA_DART_KEEP")
        dart = "config" if env is None else (None if env.lower() == "none" else float(env))
        model = load_ishinca(checkpoint_path, device=device, dtype=self.DTYPE, dart_keep=dart)
        logger.info("Ishinca loaded from %s (dart_keep=%s)", checkpoint_path, model.dart_keep)
        self._use_classifier_free_guidance_nav = False
        self._init_common(
            model=model, processor=get_processor(model), helper_module=__import__("alpamayo1_5.helper", fromlist=["x"]),
            device=torch.device(device), camera_ids=camera_ids, context_length=context_length,
            num_traj_samples=num_traj_samples, top_p=top_p, temperature=temperature,
        )
