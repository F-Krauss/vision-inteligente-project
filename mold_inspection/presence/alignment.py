"""Zone-restricted geometric alignment.

The homography is used for GEOMETRY ONLY (mapping registered piece locations
into the inspection frame). Cross-image photometry is never trusted at the
global level — experiments showed handheld re-shots saturate any global diff.

Keypoints on the golden side are restricted to the (dilated) zone polygon so
the fit models the zone plane instead of the full 3D mold; residual parallax
is absorbed per piece by context-template NCC refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from mold_inspection.presence.imageio import cap_long_side, clahe_gray

_MIN_MATCHES = 12
_RATIO_TEST = 0.78


@dataclass
class ZoneAlignment:
    ok: bool
    homography: np.ndarray | None  # maps inspection px -> golden px (at full input scale)
    inliers: int
    inlier_ratio: float
    scale: float
    rotation_deg: float
    reproj_px: float
    reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


def polygon_mask(shape: tuple[int, ...], polygon_norm: list[tuple[float, float]], dilate_frac: float = 0.0) -> np.ndarray:
    """Rasterize a normalized polygon into a uint8 mask, optionally dilated."""
    h, w = shape[:2]
    pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in polygon_norm], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    if dilate_frac > 0:
        k = max(3, int(round(dilate_frac * max(h, w))) | 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def _homography_scale_rotation(h_mat: np.ndarray) -> tuple[float, float]:
    a, b = float(h_mat[0, 0]), float(h_mat[0, 1])
    c, d = float(h_mat[1, 0]), float(h_mat[1, 1])
    det = a * d - b * c
    scale = math.sqrt(abs(det)) if det != 0 else 0.0
    rotation = math.degrees(math.atan2(c, a))
    return scale, rotation


def align_zone(
    golden_bgr: np.ndarray,
    insp_bgr: np.ndarray,
    zone_polygon_norm: list[tuple[float, float]] | None,
    *,
    max_side: int = 1600,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.25,
    max_reproj_px: float = 3.0,
    scale_bounds: tuple[float, float] = (0.5, 2.0),
    max_rotation_deg: float = 30.0,
) -> ZoneAlignment:
    """Estimate the homography mapping the inspection image into golden coordinates.

    Matching runs at a capped working resolution; the returned homography is
    rescaled to map between the ORIGINAL input resolutions.
    """
    gh, gw = golden_bgr.shape[:2]
    ih, iw = insp_bgr.shape[:2]

    golden_small = cap_long_side(golden_bgr, max_side)
    insp_small = cap_long_side(insp_bgr, max_side)
    sg = golden_small.shape[1] / gw
    si = insp_small.shape[1] / iw

    golden_gray = clahe_gray(cv2.cvtColor(golden_small, cv2.COLOR_BGR2GRAY))
    insp_gray = clahe_gray(cv2.cvtColor(insp_small, cv2.COLOR_BGR2GRAY))

    mask = None
    if zone_polygon_norm:
        mask = polygon_mask(golden_gray.shape, zone_polygon_norm, dilate_frac=0.04)

    sift = cv2.SIFT_create(nfeatures=4000)
    kp_g, des_g = sift.detectAndCompute(golden_gray, mask)
    kp_i, des_i = sift.detectAndCompute(insp_gray, None)
    if des_g is None or des_i is None or len(kp_g) < _MIN_MATCHES or len(kp_i) < _MIN_MATCHES:
        return ZoneAlignment(False, None, 0, 0.0, 0.0, 0.0, 0.0, reason="not_enough_keypoints")

    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    knn = matcher.knnMatch(des_i, des_g, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < _RATIO_TEST * n.distance]
    if len(good) < _MIN_MATCHES:
        return ZoneAlignment(False, None, 0, 0.0, 0.0, 0.0, 0.0, reason="not_enough_matches",
                             diagnostics={"matches": len(good)})

    src = np.float32([kp_i[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_g[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    h_small, inlier_mask = cv2.findHomography(src, dst, method, 3.0, maxIters=5000, confidence=0.999)
    if h_small is None or inlier_mask is None:
        return ZoneAlignment(False, None, 0, 0.0, 0.0, 0.0, 0.0, reason="homography_failed",
                             diagnostics={"matches": len(good)})

    inlier_mask = inlier_mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)

    proj = cv2.perspectiveTransform(src[inlier_mask], h_small)
    reproj_px = float(np.linalg.norm(proj - dst[inlier_mask], axis=2).mean()) if inliers else 1e9

    scale, rotation = _homography_scale_rotation(h_small)

    # Rescale: insp_full -> insp_small -> golden_small -> golden_full
    s_i = np.diag([si, si, 1.0]).astype(np.float64)
    s_g_inv = np.diag([1.0 / sg, 1.0 / sg, 1.0]).astype(np.float64)
    h_full = s_g_inv @ h_small @ s_i

    diagnostics = {"matches": len(good), "keypoints_golden": len(kp_g), "keypoints_insp": len(kp_i)}
    reason = None
    ok = True
    if inliers < min_inliers:
        ok, reason = False, "low_inliers"
    elif inlier_ratio < min_inlier_ratio:
        ok, reason = False, "low_inlier_ratio"
    elif reproj_px > max_reproj_px:
        ok, reason = False, "high_reprojection_error"
    elif not (scale_bounds[0] <= scale <= scale_bounds[1]):
        ok, reason = False, "scale_out_of_bounds"
    elif abs(rotation) > max_rotation_deg:
        ok, reason = False, "rotation_out_of_bounds"

    return ZoneAlignment(ok, h_full, inliers, inlier_ratio, scale, rotation, reproj_px,
                         reason=reason, diagnostics=diagnostics)


def warp_to_golden(insp_bgr: np.ndarray, alignment: ZoneAlignment, golden_shape: tuple[int, ...]) -> np.ndarray:
    """Warp the inspection image into the golden frame at golden resolution."""
    if alignment.homography is None:
        raise ValueError("alignment has no homography")
    gh, gw = golden_shape[:2]
    return cv2.warpPerspective(insp_bgr, alignment.homography, (gw, gh), flags=cv2.INTER_LINEAR)


def refine_piece(
    golden_gray: np.ndarray,
    insp_warped_gray: np.ndarray,
    bbox_px: tuple[int, int, int, int],
    *,
    search_frac: float = 0.03,
    context_scale: float = 1.8,
) -> tuple[int, int, float]:
    """Locally refine a piece location by matching its context ring ONLY.

    The template is the bbox expanded by ``context_scale`` with the piece
    interior MASKED OUT, so the match is driven purely by the surrounding
    plate geometry (socket rim, screw holes, edges). This keeps the lock on
    the correct socket even when the piece itself is missing — an unmasked
    template collapses (the bright piece vs dark cavity dominates the
    correlation) and a piece-only template would wander to a lookalike region
    and could turn a missing piece into a false "present".

    Returns (dx, dy, context_score) where dx/dy shift the bbox inside the
    warped inspection image and context_score in [0, 1] measures how well the
    SURROUNDINGS matched (a position-confidence value — deliberately blind to
    the piece itself; presence evidence comes from signals.py).
    """
    h, w = golden_gray.shape[:2]
    x1, y1, x2, y2 = bbox_px
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        return 0, 0, 0.0

    # Context template around the piece, clipped to the image.
    tw, th = int(bw * context_scale), int(bh * context_scale)
    tx1, ty1 = max(0, cx - tw // 2), max(0, cy - th // 2)
    tx2, ty2 = min(w, cx + tw // 2), min(h, cy + th // 2)
    template = golden_gray[ty1:ty2, tx1:tx2]
    if template.size == 0 or template.std() < 2.0:
        return 0, 0, 0.0

    # Mask: ring only — zero out the piece interior (in template coordinates).
    mask = np.full(template.shape, 255, dtype=np.uint8)
    mask[max(0, y1 - ty1):max(0, y2 - ty1), max(0, x1 - tx1):max(0, x2 - tx1)] = 0
    if int((mask > 0).sum()) < 64:
        return 0, 0, 0.0

    margin = int(round(search_frac * max(h, w)))
    sx1, sy1 = max(0, tx1 - margin), max(0, ty1 - margin)
    sx2, sy2 = min(w, tx2 + margin), min(h, ty2 + margin)
    search = insp_warped_gray[sy1:sy2, sx1:sx2]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0, 0, 0.0

    # TM_SQDIFF_NORMED is the masked-matching mode OpenCV supports reliably
    # (TM_CCOEFF_NORMED ignores masks). 0 = perfect, so score = 1 - min.
    res = cv2.matchTemplate(search, template, cv2.TM_SQDIFF_NORMED, mask=mask)
    res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
    min_val, _, loc, _ = cv2.minMaxLoc(res)
    dx = (sx1 + loc[0]) - tx1
    dy = (sy1 + loc[1]) - ty1
    # Clamp runaway shifts to the search margin (defensive).
    dx = int(np.clip(dx, -margin, margin))
    dy = int(np.clip(dy, -margin, margin))
    return dx, dy, float(np.clip(1.0 - min_val, 0.0, 1.0))
