"""Lighting-invariant per-piece presence signals.

Three families of evidence, none of which compares raw brightness across
images (the thing that breaks under auto-exposure / moving light):

1. interior-vs-ring ratio  — computed WITHIN one image; exposure and white
   balance cancel in the ratio. Compared across images only as a delta of
   ratios. Empty sockets are cavities: darker than their own rim under any
   illumination (ambient occlusion).
2. edge structure          — edge density + gradient-orientation histogram on
   CLAHE-normalized patches; a missing piece changes local structure.
3. NCC context peak        — produced by alignment.refine_piece.

Embedding cosine (embedder.py) is computed by piece_signals as well.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from mold_inspection.presence.imageio import clahe_gray

_GRAD_BINS = 18


def _clip_bbox(bbox_px: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox_px
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _expand_bbox(bbox_px: tuple[int, int, int, int], factor: float, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_px
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * factor / 2.0, (y2 - y1) * factor / 2.0
    return _clip_bbox((int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)), shape)


def interior_ring_stats(
    gray: np.ndarray,
    bbox_px: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    ring_scale: float = 1.6,
) -> dict[str, float]:
    """Median brightness of the piece interior vs its surrounding ring, same image.

    Uses RAW grayscale on purpose: CLAHE would equalize away exactly the
    cavity-darkness evidence this signal measures. The cross-image comparison
    happens on the RATIO, which is exposure-invariant.
    """
    bbox_px = _clip_bbox(bbox_px, gray.shape)
    x1, y1, x2, y2 = bbox_px
    interior = gray[y1:y2, x1:x2]
    if mask is not None:
        m = mask[y1:y2, x1:x2] > 0
        interior_vals = interior[m] if m.any() else interior.ravel()
    else:
        interior_vals = interior.ravel()

    ex1, ey1, ex2, ey2 = _expand_bbox(bbox_px, ring_scale, gray.shape)
    ring_region = gray[ey1:ey2, ex1:ex2].astype(np.float32).copy()
    ring_mask = np.ones(ring_region.shape, dtype=bool)
    ring_mask[(y1 - ey1):(y2 - ey1), (x1 - ex1):(x2 - ex1)] = False
    ring_vals = ring_region[ring_mask]
    if ring_vals.size < 16:
        ring_vals = ring_region.ravel()

    interior_med = float(np.median(interior_vals)) if interior_vals.size else 0.0
    ring_med = float(np.median(ring_vals)) if ring_vals.size else 1.0
    return {
        "interior_median": interior_med,
        "ring_median": ring_med,
        "ring_ratio": interior_med / max(ring_med, 1.0),
        "interior_std": float(np.std(interior_vals.astype(np.float32))) if interior_vals.size else 0.0,
    }


def edge_stats(gray_clahe: np.ndarray, bbox_px: tuple[int, int, int, int]) -> dict[str, Any]:
    """Edge density and gradient-orientation histogram inside the bbox."""
    x1, y1, x2, y2 = _clip_bbox(bbox_px, gray_clahe.shape)
    patch = gray_clahe[y1:y2, x1:x2]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # [0,1)

    thr = max(30.0, float(np.percentile(mag, 75)))
    strong = mag >= thr
    density = float(strong.mean())

    hist = np.zeros(_GRAD_BINS, dtype=np.float64)
    if strong.any():
        bins = np.minimum((ang[strong] * _GRAD_BINS).astype(int), _GRAD_BINS - 1)
        weights = mag[strong]
        np.add.at(hist, bins, weights)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
    return {"edge_density": density, "grad_hist": hist}


def grad_hist_cosine(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    na, nb = np.linalg.norm(hist_a), np.linalg.norm(hist_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(hist_a, hist_b) / (na * nb))


def piece_signals(
    golden_bgr: np.ndarray,
    insp_warped_bgr: np.ndarray,
    piece: dict[str, Any],
    embedder: "Any",
    *,
    golden_gray: np.ndarray | None = None,
    golden_gray_clahe: np.ndarray | None = None,
    insp_gray: np.ndarray | None = None,
    insp_gray_clahe: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all presence signals for one piece.

    ``piece`` needs ``bbox_px`` (x1,y1,x2,y2 in golden coordinates). Grayscale
    images can be passed in to avoid recomputation across many pieces.
    Returns a flat dict of scalars (plus the refined bbox for evidence).
    """
    from mold_inspection.presence.alignment import refine_piece

    if golden_gray is None:
        golden_gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
    if insp_gray is None:
        insp_gray = cv2.cvtColor(insp_warped_bgr, cv2.COLOR_BGR2GRAY)
    if golden_gray_clahe is None:
        golden_gray_clahe = clahe_gray(golden_gray)
    if insp_gray_clahe is None:
        insp_gray_clahe = clahe_gray(insp_gray)

    bbox = _clip_bbox(tuple(piece["bbox_px"]), golden_gray.shape)

    dx, dy, context_score = refine_piece(golden_gray_clahe, insp_gray_clahe, bbox)
    if context_score >= 0.30:
        x1, y1, x2, y2 = bbox
        insp_bbox = _clip_bbox((x1 + dx, y1 + dy, x2 + dx, y2 + dy), insp_gray.shape)
    else:
        insp_bbox = bbox  # context match failed; stay at the aligned location

    g_ring = interior_ring_stats(golden_gray, bbox)
    i_ring = interior_ring_stats(insp_gray, insp_bbox)

    g_edges = edge_stats(golden_gray_clahe, bbox)
    i_edges = edge_stats(insp_gray_clahe, insp_bbox)

    # Interior correlation: presence evidence proper. Same-size CLAHE'd
    # interior patches compared directly at the refined location.
    g_interior = golden_gray_clahe[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
    i_interior = insp_gray_clahe[insp_bbox[1]:insp_bbox[3], insp_bbox[0]:insp_bbox[2]].astype(np.float32)
    if i_interior.shape != g_interior.shape and g_interior.size > 0:
        i_interior = cv2.resize(i_interior, (g_interior.shape[1], g_interior.shape[0]))
    interior_ncc = 0.0
    if g_interior.size > 16 and g_interior.std() > 1e-3 and i_interior.std() > 1e-3:
        interior_ncc = float(np.corrcoef(g_interior.ravel(), i_interior.ravel())[0, 1])
        if np.isnan(interior_ncc):
            interior_ncc = 0.0

    emb_pad = 1.25
    g_patch_box = _expand_bbox(bbox, emb_pad, golden_bgr.shape)
    i_patch_box = _expand_bbox(insp_bbox, emb_pad, insp_warped_bgr.shape)
    g_patch = golden_bgr[g_patch_box[1]:g_patch_box[3], g_patch_box[0]:g_patch_box[2]]
    i_patch = insp_warped_bgr[i_patch_box[1]:i_patch_box[3], i_patch_box[0]:i_patch_box[2]]
    embs = embedder.embed([g_patch, i_patch])
    emb_cos = float(np.dot(embs[0], embs[1]))

    return {
        "emb_cos": emb_cos,
        "interior_ncc": interior_ncc,
        "context_score": context_score,
        "ring_ratio": i_ring["ring_ratio"],
        "ring_ratio_golden": g_ring["ring_ratio"],
        "edge_density": i_edges["edge_density"],
        "edge_density_golden": g_edges["edge_density"],
        "grad_cos": grad_hist_cosine(g_edges["grad_hist"], i_edges["grad_hist"]),
        "refined_dx": dx,
        "refined_dy": dy,
        "insp_bbox_px": list(insp_bbox),
    }
