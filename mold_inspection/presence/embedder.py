"""Patch embedding backends for appearance comparison.

Two interchangeable backbones, selected by Phase 0 benchmark (ROC vs latency):

- ``dinov2_vits14_reg``  — DINOv2-S with registers (384-d). Self-supervised
  features are markedly more robust to illumination/specularity than
  ImageNet-classifier features; the established choice for one-shot
  industrial anomaly detection. ~85MB, ~30-80ms/patch on CPU.
- ``mobilenet_v3_large`` — torchvision penultimate features (960-d). 5-10x
  faster, zero extra downloads beyond torchvision weights.

Patches are CLAHE-normalized (L channel) before embedding so the model sees
contrast-stabilized inputs under lighting drift.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np

from mold_inspection.presence.imageio import clahe_bgr

if TYPE_CHECKING:  # pragma: no cover
    import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_INPUT_SIZE = 224

_DEFAULT_NAME = os.environ.get("MOLD_PRESENCE_EMBEDDER", "mobilenet_v3_large")

_lock = threading.Lock()
_instances: dict[str, "PatchEmbedder"] = {}


class PatchEmbedder:
    """Batched, L2-normalized patch embeddings on CPU (or CUDA when available)."""

    def __init__(self, name: str = _DEFAULT_NAME, device: str | None = None) -> None:
        import torch

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._load(name).to(self.device).eval()

    @staticmethod
    def _load(name: str):
        import torch

        if name == "dinov2_vits14_reg":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False)
        if name == "mobilenet_v3_large":
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            return _MobileNetFeatures(model)
        raise ValueError(f"Unknown embedder: {name}")

    def _preprocess(self, patches_bgr: list[np.ndarray]) -> "torch.Tensor":
        import torch

        batch = np.empty((len(patches_bgr), 3, _INPUT_SIZE, _INPUT_SIZE), dtype=np.float32)
        for i, patch in enumerate(patches_bgr):
            if patch.size == 0:
                patch = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
            patch = clahe_bgr(patch)
            patch = cv2.resize(patch, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            batch[i] = rgb.transpose(2, 0, 1)
        return torch.from_numpy(batch).to(self.device)

    def embed(self, patches_bgr: list[np.ndarray]) -> np.ndarray:
        """Return [N, D] float32, L2-normalized rows."""
        import torch

        if not patches_bgr:
            return np.empty((0, 0), dtype=np.float32)
        with torch.no_grad():
            feats = self._model(self._preprocess(patches_bgr))
        feats = feats.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms


class _MobileNetFeatures:
    """Penultimate-feature extractor wrapper (960-d global average pool)."""

    def __init__(self, model) -> None:
        self._features = model.features
        self._pool = model.avgpool

    def to(self, device):
        self._features = self._features.to(device)
        return self

    def eval(self):
        self._features.eval()
        return self

    def __call__(self, x):
        import torch

        y = self._features(x)
        y = self._pool(y)
        return torch.flatten(y, 1)


def get_embedder(name: str | None = None) -> PatchEmbedder:
    """Process-wide singleton per backend name."""
    resolved = name or _DEFAULT_NAME
    with _lock:
        if resolved not in _instances:
            _instances[resolved] = PatchEmbedder(resolved)
        return _instances[resolved]
