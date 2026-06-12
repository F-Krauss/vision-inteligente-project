"""Image loading helpers shared by the presence modules and offline scripts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Read an image (JPEG/PNG/HEIC) as BGR, optionally capping the long side."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        try:
            import pillow_heif
            from PIL import Image

            pillow_heif.register_heif_opener()
            with Image.open(path) as pil:
                img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not read image: {path}") from exc
    if max_side is not None:
        img = cap_long_side(img, max_side)
    return img


def cap_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def clahe_gray(gray: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    """Local-contrast normalized grayscale; stabilizes matching under lighting drift."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def clahe_bgr(bgr: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel only, preserving color for embedding models."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
