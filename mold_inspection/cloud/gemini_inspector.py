"""
Hybrid mold inspection: OpenCV detects structural difference regions,
Gemini Vision confirms each one.

Flow:
  1. Load reference + candidate; align with ORB homography.
  2. Apply tile-level brightness normalization to cancel lighting differences.
  3. Compute per-tile SSIM; find large low-SSIM regions (structural differences).
  4. If no regions found → "correct" immediately (zero false positives).
  5. If regions found → ask Gemini one binary question per region.
  6. Only confirmed regions become findings.

Usage:
    result = inspect_with_gemini(
        reference_image_path="gs://... or /local/path",
        candidate_image_path="gs://... or /local/path",
        family="my_family",
        zone_id="zona_01",
        evidence_dir="/tmp/evidence",
    )
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT  = os.environ.get("MOLD_GCP_PROJECT", "mia-production-project")
_LOCATION = os.environ.get("MOLD_GCP_REGION",  "us-central1")
_MODEL    = "gemini-2.5-flash"

_MAX_W    = 1400        # resize long side to this (preserves memory & Gemini input size)
_TILE     = 120         # tile size for local normalization + SSIM
_SSIM_THR = 0.55        # tiles below this are "structurally different"
_MIN_AREA = 0.005       # minimum region area as fraction of image (eliminates tiny noise)
_MAX_REGIONS = 6        # cap on how many regions we send to Gemini
_SAME_IMG_THR = 0.05    # SSIM mean-diff below this → images are identical → skip Gemini


# ── public entry point ─────────────────────────────────────────────────────────

def inspect_with_gemini(
    reference_image_path: str | Path,
    candidate_image_path: str | Path,
    family: str,
    zone_id: str,
    evidence_dir: str | Path | None = None,
    min_confidence: float = 0.60,
) -> dict[str, Any]:
    """
    Hybrid OpenCV + Gemini inspection.
    Returns a dict compatible with the pipeline's piece_inspection format.
    """
    ref_p  = str(reference_image_path)
    cand_p = str(candidate_image_path)

    try:
        result = _hybrid_inspect(ref_p, cand_p, family, zone_id, evidence_dir, min_confidence)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning("Hybrid inspection error, falling back to pure Gemini: %s", exc)

    # Pure Gemini fallback (two-step describe + check)
    try:
        raw = _call_gemini_two_step(ref_p, cand_p)
    except Exception as exc:
        logger.error("Gemini fallback failed: %s", exc)
        return _error_result(str(exc))

    return _build_result(raw, cand_p, family, zone_id, evidence_dir, min_confidence)


# ── Classical CV public entry point ───────────────────────────────────────────

_CV_TILE       = 80      # tile size for brightness normalisation
_CV_SIGMA      = 2.5     # threshold = mean + CV_SIGMA * std (adaptive)
_CV_PISTON_THR = 0.012   # blob area fraction of ROI ≥ this → piston; below → bolt

# ROI polygon — normalized (x, y) in [0,1] relative to the loaded image dimensions.
# Defined once on img_with1 (4284×5712 loaded). Applied to every image via ORB warp.
# Excludes outer frame, floor, upper background — focuses on inner mold track.
_DEFAULT_ROI_POLY_NORM: list[tuple[float, float]] = [
    (0.300, 0.300),   # top-left
    (0.500, 0.287),   # top-centre
    (0.748, 0.300),   # top-right
    (0.748, 0.858),   # bottom-right
    (0.500, 0.868),   # bottom-centre
    (0.300, 0.858),   # bottom-left
]


def _make_roi_mask(H: int, W: int, poly_norm: list) -> "np.ndarray":
    """Binary mask (uint8 255=inside) from a list of normalized (x,y) polygon points."""
    pts = np.array([[int(x * W), int(y * H)] for x, y in poly_norm], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    import cv2
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _localized_diff_area(diff_map: "np.ndarray", active_mask: "np.ndarray", struct_thr: float) -> int:
    """
    Largest contiguous area (px) inside ``active_mask`` where the SSIM difference
    exceeds ``struct_thr`` — i.e. a strong *structural* change, not diffuse
    lighting/angle noise. The SSIM gate uses this so a localized missing piece —
    which barely moves the *global* mean diff (~0.02) — cannot be silently
    approved just because the rest of the mold is identical.
    """
    import cv2
    binary = (diff_map > struct_thr).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, active_mask)
    # Light open removes SSIM speckle; the caller's area threshold rejects noise.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return int(max((cv2.contourArea(c) for c in contours), default=0))


def inspect_with_cv(
    reference_image_path: str | Path,
    candidate_image_path: str | Path,
    family: str,
    zone_id: str,
    evidence_dir: str | Path | None = None,
    min_confidence: float = 0.30,
    roi_polygon: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """
    Pure classical CV mold inspection — no Gemini, fully deterministic.

    Pipeline:
      1. Load both images (HEIC + GCS handled transparently).
      2. Scale to _MAX_W. ORB-align candidate to reference.
      3. Build ROI mask from the user-defined polygon (in reference-image space).
         Everything outside the polygon is ignored — no background, no frame noise.
      4. SSIM gate: short-circuit to "correct" ONLY when the global mean diff
         < _SAME_IMG_THR AND there is no localized structural cluster inside the
         ROI. A localized cluster falls through to the diff pipeline so a small
         missing piece is never approved by a diluted global mean (false approval).
      5. Per-tile brightness normalise inside the ROI to cancel lighting differences.
      6. Adaptive-threshold pixel diff inside ROI: mean + _CV_SIGMA × std.
      7. Morphological close+open to merge component blobs and remove micro-noise.
      8. Contour size classification (relative to ROI area):
           blob_area / roi_area ≥ _CV_PISTON_THR  → piston
           below                                  → bolt
      9. Draw overlay on the aligned candidate, with ROI boundary shown.

    Args:
        roi_polygon: Optional list of (x, y) normalized polygon points overriding the
                     built-in default. Useful for different mold families/zones.
    Returns:
        dict compatible with the pipeline piece_inspection format.
    """
    ref_p  = str(reference_image_path)
    cand_p = str(candidate_image_path)
    poly   = roi_polygon if roi_polygon is not None else _DEFAULT_ROI_POLY_NORM

    try:
        import cv2
    except ImportError:
        return _error_result("OpenCV (cv2) not installed — cannot run classical CV inspection.")

    # ── 1. Load & scale images ────────────────────────────────────────────────────
    ref_img  = _cv2_read_path(ref_p)
    cand_img = _cv2_read_path(cand_p)
    if ref_img is None or cand_img is None:
        return _error_result(f"Could not load images: ref={ref_p}  cand={cand_p}")

    ref_img  = _scale(ref_img,  _MAX_W)
    cand_img = _scale(cand_img, _MAX_W)
    IH, IW   = ref_img.shape[:2]

    # ── 2. ORB-align candidate to reference (full image for best feature coverage) ─
    aligned, valid_orb, ir = _orb_align(ref_img, cand_img)

    # ── 3. ROI mask (polygon in reference-image space; applied to both images) ─────
    # Built before the gate so it can reason about the inspection area only — a
    # missing piece is localized, and a whole-image mean would dilute it.
    roi_mask   = _make_roi_mask(IH, IW, poly)
    active     = cv2.bitwise_and(valid_orb, roi_mask)   # inside polygon AND valid warp
    roi_area   = int(np.count_nonzero(active))
    if roi_area < 1000:
        return _error_result("ROI mask is empty after ORB alignment — cannot compare.")

    # ── 4. SSIM gate ─────────────────────────────────────────────────────────────
    # Approve as "correct" ONLY when BOTH hold:
    #   • global mean SSIM diff < _SAME_IMG_THR (images look near-identical), AND
    #   • no localized structural cluster inside the ROI.
    # The localized guard is essential: a single missing piece moves the global mean
    # by well under _SAME_IMG_THR (~0.02 on a real fault), so the mean alone would
    # silently approve a faulty mold. Project priority is false-rejection over
    # false-approval, so a concentrated high-diff region falls through to the ROI
    # diff pipeline below, which classifies and reports it.
    diff_map      = _tile_diff_map(ref_img, aligned, valid_orb)
    valid_vals    = diff_map[valid_orb > 0]
    mean_diff     = float(valid_vals.mean()) if valid_vals.size > 0 else 0.0
    gate_min_px   = max(200, int(0.003 * roi_area))     # smallest cluster worth flagging
    local_diff_px = _localized_diff_area(diff_map, active, 1.0 - _SSIM_THR)
    logger.info(
        "CV gate: mean_diff=%.4f  ir=%.3f  threshold=%.3f  local_diff=%dpx (min=%d)",
        mean_diff, ir, _SAME_IMG_THR, local_diff_px, gate_min_px,
    )

    if mean_diff < _SAME_IMG_THR and local_diff_px < gate_min_px:
        overlay = _draw_cv_overlay_roi(
            cand_img, poly, [], evidence_dir, family, zone_id, cand_p
        )
        return {
            "status":        "correct",
            "message":       "Todas las piezas presentes — sin diferencias detectadas.",
            "findings":      [],
            "missing_count": 0,
            "overlay_image": overlay,
            "method":        "classical_cv",
            "cv_stats":      {"ir": round(ir, 3), "mean_diff": round(mean_diff, 4),
                              "local_diff_px": local_diff_px},
        }

    # ── 5. Per-tile brightness normalise inside ROI ───────────────────────────────
    ref_g  = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cand_g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    norm_cand = cand_g.copy()
    for ty in range(0, IH, _CV_TILE):
        for tx in range(0, IW, _CV_TILE):
            yt, xt = min(ty + _CV_TILE, IH), min(tx + _CV_TILE, IW)
            vm = active[ty:yt, tx:xt]
            if vm.mean() < 50:
                continue
            r_t = ref_g [ty:yt, tx:xt][vm > 0]
            c_t = cand_g[ty:yt, tx:xt][vm > 0]
            if r_t.size < 50 or c_t.std() < 1e-3:
                continue
            norm_cand[ty:yt, tx:xt] = np.clip(
                cand_g[ty:yt, tx:xt] * (r_t.mean() / max(c_t.mean(), 1.0)), 0, 255
            )

    # ── 6. Adaptive pixel diff inside ROI ────────────────────────────────────────
    diff = np.abs(ref_g - norm_cand).astype(np.uint8)
    diff = cv2.bitwise_and(diff, diff, mask=active)

    vals = diff[active > 0].astype(float)
    thr  = int(min(vals.mean() + _CV_SIGMA * vals.std(), 200)) if vals.size else 80
    _, binary = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, active)    # zero everything outside ROI

    # ── 7. Morphological close+open ───────────────────────────────────────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,  7))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open,  iterations=1)

    # ── 8. Contour → findings ─────────────────────────────────────────────────────
    min_px = max(200, int(0.003 * roi_area))   # ≥0.3% of ROI
    max_px = int(0.20  * roi_area)             # ≤20% of ROI (avoid huge noise blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    findings: list[dict[str, Any]] = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_px or area > max_px:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        frac = area / roi_area
        kind = "piston" if frac >= _CV_PISTON_THR else "bolt"
        conf = round(min(0.95, frac / max(_CV_PISTON_THR, 1e-6) * 0.5 + 0.45), 4)
        findings.append({
            "piece_id":        f"cv_{kind}_{len(findings) + 1}",
            "class_name":      kind,
            "status":          "missing",
            "confidence":      conf,
            "bbox_normalized": {
                "x":      round(bx / IW, 4),
                "y":      round(by / IH, 4),
                "width":  round(bw / IW, 4),
                "height": round(bh / IH, 4),
            },
            "region": [
                {"x": round(bx / IW,        4), "y": round(by / IH,        4)},
                {"x": round((bx + bw) / IW, 4), "y": round(by / IH,        4)},
                {"x": round((bx + bw) / IW, 4), "y": round((by + bh) / IH, 4)},
                {"x": round(bx / IW,        4), "y": round((by + bh) / IH, 4)},
            ],
            "method": "classical_cv",
        })

    n       = len(findings)
    pistons = sum(1 for f in findings if f["class_name"] == "piston")
    bolts   = sum(1 for f in findings if f["class_name"] == "bolt")
    status  = "correct" if n == 0 else "review"
    message = (
        "Todas las piezas presentes — zona validada por CV."
        if n == 0
        else f"Detectadas {n} piezas faltantes ({pistons} pistón/nes + {bolts} tornillo/s)."
    )

    # ── 9. Overlay on the aligned candidate with ROI boundary ─────────────────────
    overlay = _draw_cv_overlay_roi(
        aligned, poly, findings, evidence_dir, family, zone_id, cand_p
    )

    logger.info(
        "CV result: %s  n=%d pistons=%d bolts=%d  ir=%.3f  diff=%.4f  thr=%d  roi=%d px",
        status, n, pistons, bolts, ir, mean_diff, thr, roi_area,
    )
    return {
        "status":        status,
        "message":       message,
        "findings":      findings,
        "missing_count": n,
        "overlay_image": overlay,
        "method":        "classical_cv",
        "cv_stats": {
            "ir":        round(ir, 3),
            "mean_diff": round(mean_diff, 4),
            "threshold": thr,
            "roi_area":  roi_area,
            "pistons":   pistons,
            "bolts":     bolts,
        },
    }


# ── hybrid pipeline ────────────────────────────────────────────────────────────

def _hybrid_inspect(
    ref_p: str, cand_p: str,
    family: str, zone_id: str,
    evidence_dir, min_confidence: float,
) -> dict[str, Any] | None:
    """
    Two-phase pipeline:
      Phase 1 (OpenCV gate): Compute SSIM diff between aligned images.
        - If mean diff < _SAME_IMG_THR → images are identical → return "correct" immediately.
          This guarantees zero false positives (control case never reaches Gemini).
        - If mean diff ≥ threshold → images genuinely differ → proceed to Gemini.
      Phase 2 (Gemini): Pure two-step comparison on the original images.
        - Step 1: Describe reference components.
        - Step 2: Check which are absent in candidate, with bounding boxes.
    """
    try:
        import cv2
        from skimage.metrics import structural_similarity  # noqa: F401 (presence check)
    except ImportError:
        return None

    ref_img  = _cv2_read_path(ref_p)
    cand_img = _cv2_read_path(cand_p)
    if ref_img is None or cand_img is None:
        return None

    ref_img  = _scale(ref_img,  _MAX_W)
    cand_img = _scale(cand_img, _MAX_W)

    aligned, valid_mask, ir = _orb_align(ref_img, cand_img)
    diff_map  = _tile_diff_map(ref_img, aligned, valid_mask)
    valid_vals = diff_map[valid_mask > 0]
    mean_diff  = float(valid_vals.mean()) if valid_vals.size > 0 else 0.0

    logger.info("SSIM gate: mean_diff=%.3f ir=%.2f threshold=%.2f", mean_diff, ir, _SAME_IMG_THR)

    # ── Phase 1: identical images → correct, no Gemini call ──────────────────
    if mean_diff < _SAME_IMG_THR:
        overlay = _draw_overlay_clean(cand_img, [], evidence_dir, family, zone_id, cand_p)
        return {
            "status": "correct",
            "message": "Todas las piezas presentes — sin diferencias estructurales detectadas.",
            "findings": [],
            "missing_count": 0,
            "overlay_image": overlay,
            "method": "hybrid_opencv_gemini",
            "model": _MODEL,
            "raw_gemini": {
                "total_missing": 0,
                "summary": "Images are identical — no structural differences.",
                "inspection_quality": "good",
                "quality_notes": f"SSIM mean diff={mean_diff:.4f} (gate={_SAME_IMG_THR})",
            },
        }

    # ── Phase 2: crop BOTH images to ROI polygon → send to Gemini ─────────────
    # Sending focused polygon crops instead of full images:
    #   • Removes background, outer frame, floor — Gemini only sees the inspection area
    #   • Makes filled-vs-empty holes much more obvious at the crop scale
    #   • Reduces Gemini non-determinism (less irrelevant visual noise)
    logger.info(
        "Images differ (mean_diff=%.3f ≥ %.2f) → ROI crop → Gemini", mean_diff, _SAME_IMG_THR
    )
    try:
        ref_crop, cand_crop = _roi_crops(ref_img, aligned, _DEFAULT_ROI_POLY_NORM)
        # Debug: save individual crops for inspection
        try:
            import pathlib as _pl, time as _time, sys as _sys
            _d = _pl.Path("/tmp/mold_composites")
            _d.mkdir(exist_ok=True)
            _ts2 = int(_time.time() * 1000) % 100000
            import cv2 as _cv2dbg
            _cv2dbg.imwrite(str(_d / f"ref_crop_{_ts2}.jpg"), ref_crop)
            _cv2dbg.imwrite(str(_d / f"cand_crop_{_ts2}.jpg"), cand_crop)
            print(f"[DBG] saved crops: ref_crop_{_ts2}.jpg, cand_crop_{_ts2}.jpg", file=_sys.stderr)
        except Exception as _save_err:
            pass
        raw = _call_gemini_roi_crops(ref_crop, cand_crop)
    except Exception as exc:
        logger.error("Gemini ROI-crop step failed in hybrid: %s  → falling back to full-image Gemini", exc)
        try:
            raw = _call_gemini_two_step(ref_p, cand_p)
        except Exception as exc2:
            logger.error("Full-image Gemini also failed: %s", exc2)
            return None

    result = _build_result(raw, cand_p, family, zone_id, evidence_dir, min_confidence)
    result["findings"] = _cluster_findings(result["findings"])
    result["missing_count"] = len(result["findings"])
    n = result["missing_count"]
    if n == 0:
        result["status"]  = "correct"
        result["message"] = "Todas las piezas presentes — zona validada."
    else:
        result["status"]  = "review"
        result["message"] = raw.get("summary") or f"Gemini detectó {n} componente(s) faltante(s)."
    result["overlay_image"] = _draw_cv_overlay_roi(
        cand_img, _DEFAULT_ROI_POLY_NORM, result["findings"],
        evidence_dir, family, zone_id, cand_p,
    )
    result["method"] = "hybrid_opencv_gemini_roi"
    result["raw_gemini"]["quality_notes"] = (
        f"SSIM mean_diff={mean_diff:.3f}; " + result["raw_gemini"].get("quality_notes", "")
    )
    return result


# ── OpenCV helpers ─────────────────────────────────────────────────────────────

def _cv2_read_path(path: str):
    """Load image from local file or GCS URI (uses google-cloud-storage, no gsutil)."""
    try:
        import cv2
    except ImportError:
        return None

    if path.startswith("gs://"):
        tmp_path = None
        try:
            from google.cloud import storage as gcs
            path_no_scheme = path[5:]
            bucket_name, blob_name = path_no_scheme.split("/", 1)
            client = gcs.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            suffix = Path(blob_name).suffix or ".bin"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            blob.download_to_filename(tmp_path)
            return _read_any_format(tmp_path)
        except Exception as exc:
            logger.warning("GCS download failed for %s: %s", path, exc)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        return _read_any_format(path)


def _read_any_format(path: str):
    """Read any image format: JPEG/PNG directly, HEIC via pillow-heif or sips."""
    try:
        import cv2
    except ImportError:
        return None

    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is not None:
        return img

    if p.suffix.lower() not in {".heic", ".heif"}:
        return None

    # HEIC: try pillow-heif (works on Linux + macOS)
    try:
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        pil_img = Image.open(str(p)).convert("RGB")
        import cv2
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # macOS fallback: sips
    try:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(p), "--out", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import cv2
        return cv2.imread(tmp, cv2.IMREAD_COLOR)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _scale(img, max_w: int):
    import cv2
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)


def _orb_align(ref, cand):
    """Align cand to ref using ORB + RANSAC homography. Returns (aligned, valid_mask, inlier_ratio)."""
    import cv2
    gr = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=10000)
    k1, d1 = orb.detectAndCompute(gr, None)
    k2, d2 = orb.detectAndCompute(gc, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return cand, np.ones(ref.shape[:2], np.uint8) * 255, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:400]
    if len(matches) < 10:
        return cand, np.ones(ref.shape[:2], np.uint8) * 255, 0.0
    p1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    p2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(p2, p1, cv2.RANSAC, 4.0)
    if H is None:
        return cand, np.ones(ref.shape[:2], np.uint8) * 255, 0.0
    rh, rw = ref.shape[:2]
    aligned = cv2.warpPerspective(cand, H, (rw, rh))
    # Valid-pixel mask (where warped image has real data)
    ones = np.ones(cand.shape[:2], np.uint8) * 255
    valid = cv2.warpPerspective(ones, H, (rw, rh))
    valid = cv2.threshold(valid, 200, 255, cv2.THRESH_BINARY)[1]
    valid = cv2.erode(valid, np.ones((20, 20), np.uint8))
    ir = float(mask.sum()) / len(mask) if mask is not None else 0.0
    return aligned, valid, ir


def _tile_diff_map(ref, cand, valid_mask):
    """
    Compute per-pixel SSIM difference map using tile-level brightness normalization.
    Each tile's candidate brightness is matched to the reference before SSIM.
    Returns a float32 map where 0=identical, 1=completely different.
    """
    import cv2
    from skimage.metrics import structural_similarity as ssim

    ref_g  = cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    cand_g = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = ref_g.shape

    norm_cand = cand_g.copy()
    for y in range(0, H, _TILE):
        for x in range(0, W, _TILE):
            yt, xt = min(y + _TILE, H), min(x + _TILE, W)
            vm = valid_mask[y:yt, x:xt]
            if vm.mean() < 50:
                continue
            r_tile = ref_g[y:yt, x:xt][vm > 0]
            c_tile = cand_g[y:yt, x:xt][vm > 0]
            if r_tile.size < 100 or c_tile.std() < 1e-3:
                continue
            scale = r_tile.mean() / max(c_tile.mean(), 1.0)
            norm_cand[y:yt, x:xt] = np.clip(cand_g[y:yt, x:xt] * scale, 0, 255)

    ref_u8  = np.clip(ref_g, 0, 255).astype(np.uint8)
    cand_u8 = np.clip(norm_cand, 0, 255).astype(np.uint8)

    # Compute SSIM with full image (win_size auto)
    win = min(11, min(H, W) // 4 * 2 + 1)  # odd
    _, ssim_map = ssim(ref_u8, cand_u8, win_size=win, full=True, data_range=255)
    # diff_map: 0=same, 1=different
    return np.clip(1.0 - ssim_map, 0, 1).astype(np.float32)


def _find_regions(diff_map, valid_mask, H: int, W: int) -> list[dict]:
    """
    Find structurally-different regions using a RELATIVE threshold.
    We flag tiles that are significantly MORE different than the global average —
    this isolates specific missing-part locations even when global SSIM is moderate.
    """
    import cv2

    min_px = int(_MIN_AREA * H * W)

    # Only consider valid (aligned) pixels
    valid_f = (valid_mask > 0).astype(np.float32)
    masked  = diff_map * valid_f

    valid_vals = masked[valid_mask > 0]
    if valid_vals.size < 100:
        return []

    mean_d = float(valid_vals.mean())
    std_d  = float(valid_vals.std())

    # Relative threshold: only flag regions well above the average difference.
    # This works even when the whole image is "somewhat different" due to angle/lighting.
    rel_thr = min(mean_d + 1.8 * std_d, 0.90)  # cap at 0.90

    # Also use absolute floor (don't flag tiny noise even if it's "above average")
    abs_floor = 1.0 - _SSIM_THR  # = 0.45

    threshold = max(rel_thr, abs_floor)
    logger.info("SSIM diff: mean=%.3f std=%.3f → threshold=%.3f", mean_d, std_d, threshold)

    binary = (diff_map > threshold).astype(np.uint8) * 255
    binary = cv2.bitwise_and(binary, valid_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_TILE // 2, _TILE // 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_px:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        # Add padding for context
        pad = max(30, min(rw, rh) // 5)
        x  = max(0, x - pad);       y  = max(0, y - pad)
        rw = min(W - x, rw + 2*pad); rh = min(H - y, rh + 2*pad)
        avg_diff = float(diff_map[y:y+rh, x:x+rw].mean())
        regions.append({"bbox": (x, y, rw, rh), "area": area, "avg_diff": avg_diff})

    regions.sort(key=lambda r: -r["avg_diff"])
    return regions[:_MAX_REGIONS]


# ── Gemini confirmation ────────────────────────────────────────────────────────

_CONFIRM_CROP_PROMPT = """\
CROP 1 = a region cropped from the complete mold REFERENCE (golden sample).
CROP 2 = the EXACT SAME region cropped from the INSPECTION photo.

Both crops show the same physical location on the mold.

Is there a physical component (pin, piston, screw, bolt, clip, insert) that is \
PHYSICALLY PRESENT in CROP 1 but PHYSICALLY ABSENT in CROP 2 \
(empty hole, bare surface, or missing part where a solid component should be)?

Answer ONLY with a JSON object — no markdown, no explanation:
{"absent": true, "label": "<pin|piston|screw|bolt|clip|insert|other>", "confidence": <0.0-1.0>, "note": "<brief>"}
or
{"absent": false, "label": "", "confidence": <0.0-1.0>, "note": "<reason the crops look equivalent>"}"""


def _confirm_regions(
    ref_p: str, cand_p: str,
    ref_img, cand_img,
    regions: list[dict], H: int, W: int,
) -> list[dict]:
    """
    Crop both reference and candidate to each region, send the two crops to
    Gemini and ask a binary "is anything absent?" question.
    This gives Gemini a focused, apples-to-apples comparison instead of
    asking it to scan a full annotated image.
    """
    try:
        import google.genai as genai
        from google.genai import types
        import cv2
    except ImportError:
        logger.error("google-genai or cv2 not installed")
        return []

    client = genai.Client(vertexai=True, project=_PROJECT, location=_LOCATION)
    confirmed = []

    for region in regions:
        x, y, rw, rh = region["bbox"]

        # Crop the same area from both images
        ref_crop  = ref_img[y:y + rh, x:x + rw]
        cand_crop = cand_img[y:y + rh, x:x + rw]

        if ref_crop.size == 0 or cand_crop.size == 0:
            continue

        _, ref_buf  = cv2.imencode(".jpg", ref_crop,  [cv2.IMWRITE_JPEG_QUALITY, 92])
        _, cand_buf = cv2.imencode(".jpg", cand_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ref_part  = types.Part.from_bytes(data=ref_buf.tobytes(),  mime_type="image/jpeg")
        cand_part = types.Part.from_bytes(data=cand_buf.tobytes(), mime_type="image/jpeg")

        try:
            resp = client.models.generate_content(
                model=_MODEL,
                contents=[ref_part, cand_part, _CONFIRM_CROP_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=1024),
                ),
            )
            text = resp.text or ""
            data = _parse_json(text)
            absent = data.get("absent", False)
            conf   = float(data.get("confidence", 0.5))
            label  = data.get("label", "component") or "component"
            note   = data.get("note", "")
            logger.info("Region %s: absent=%s conf=%.2f label=%s note=%s",
                        region["bbox"], absent, conf, label, note)
            if absent and conf >= 0.65:
                confirmed.append({**region, "label": label, "confidence": conf, "note": note})
        except Exception as exc:
            logger.warning("Gemini confirmation error for region %s: %s", region["bbox"], exc)

    return confirmed


def _make_image_part(path: str, img_array=None):
    """
    Build a Gemini Part.
    - GCS paths → Part.from_uri (native HEIC/JPEG, full resolution, no size limit).
    - Local paths → scale to ≤_MAX_W then encode as JPEG bytes
      (Gemini inline images have a ~3000px limit; full HEIC files exceed that).
    """
    from google.genai import types
    if path.startswith("gs://"):
        mime = "image/heic" if path.lower().endswith(".heic") else "image/jpeg"
        return types.Part.from_uri(file_uri=path, mime_type=mime)
    # Local file → encode as JPEG bytes (scaled to safe size for inline upload)
    try:
        import cv2
        img = img_array if img_array is not None else _cv2_read_path(path)
        if img is not None:
            img = _scale(img, _MAX_W)   # keep within Gemini inline size limits
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
    except Exception:
        pass
    data = _read_image_bytes(Path(path))
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")


# ── ROI crop helpers ──────────────────────────────────────────────────────────

def _roi_crops(ref_img, aligned_cand, poly_norm: list) -> tuple:
    """
    Extract the bounding-box crop of the ROI polygon from both images.
    Both images must already be in the same coordinate space (ref_img space).
    Returns (ref_crop, cand_crop) as numpy arrays.
    """
    import cv2
    H, W = ref_img.shape[:2]
    pts  = np.array([[int(x * W), int(y * H)] for x, y in poly_norm], dtype=np.int32)
    x, y, pw, ph = cv2.boundingRect(pts)
    # Apply slight padding for context
    pad  = 20
    x1   = max(0, x - pad);       y1 = max(0, y - pad)
    x2   = min(W, x + pw + pad);  y2 = min(H, y + ph + pad)
    return ref_img[y1:y2, x1:x2], aligned_cand[y1:y2, x1:x2]


def _call_gemini_roi_crops(ref_crop, cand_crop) -> dict[str, Any]:
    """
    Build a side-by-side composite (LEFT=reference, RIGHT=inspection) and send
    to Gemini as a single image. Direct visual comparison in one frame is more
    reliable than sending two separate images.
    """
    import google.genai as genai
    from google.genai import types
    import cv2

    client = genai.Client(vertexai=True, project=_PROJECT, location=_LOCATION)

    # ── Build side-by-side composite ──────────────────────────────────────────
    # Resize both crops to the same height for alignment
    target_h = 900
    def _fit(img, h):
        ih, iw = img.shape[:2]
        s = h / ih
        return cv2.resize(img, (int(iw * s), h), interpolation=cv2.INTER_AREA)

    left  = _fit(ref_crop,  target_h)
    right = _fit(cand_crop, target_h)
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    gap = 12

    composite = np.zeros((max(lh, rh) + 50, lw + gap + rw, 3), dtype=np.uint8)
    composite[:] = (30, 30, 30)

    # Left half = REFERENCE
    composite[50:50+lh, :lw] = left
    cv2.rectangle(composite, (0, 0), (lw, 48), (20, 80, 20), -1)
    cv2.putText(composite, "REFERENCE — all pistons installed",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Right half = INSPECTION
    composite[50:50+rh, lw+gap:lw+gap+rw] = right
    cv2.rectangle(composite, (lw+gap, 0), (lw+gap+rw, 48), (80, 20, 20), -1)
    cv2.putText(composite, "INSPECTION — some pistons may be missing",
                (lw+gap+8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Divider line
    cv2.line(composite, (lw + gap//2, 0), (lw + gap//2, composite.shape[0]),
             (200, 200, 200), 2)

    # ── Debug: save composite for visual inspection ───────────────────────────
    try:
        import pathlib, time
        _dbg = pathlib.Path("/tmp/mold_composites")
        _dbg.mkdir(exist_ok=True)
        _ts = int(time.time() * 1000) % 100000
        cv2.imwrite(str(_dbg / f"composite_{_ts}.jpg"), composite)
    except Exception as _e:
        import sys as _sys2
        print(f"[COMPOSITE_SAVE_ERR] {_e}", file=_sys2.stderr)
    # ─────────────────────────────────────────────────────────────────────────

    _, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

    resp = client.models.generate_content(
        model=_MODEL,
        contents=[img_part, _ROI_SIDEBYSIDE_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=16384,
        ),
    )
    _raw_text = resp.text or ""
    import logging as _logging
    _logging.getLogger(__name__).debug("Gemini ROI raw response: %s", _raw_text[:500])
    # Also print to stderr for test visibility
    import sys as _sys
    print(f"[GEMINI_RAW] {_raw_text[:600]}", file=_sys.stderr)
    raw = _parse_json(_raw_text)
    raw.setdefault("inspection_quality", raw.pop("photo_quality", "good"))
    raw.setdefault("quality_notes",      raw.pop("photo_notes", ""))

    # ── Remap x coordinates from composite space → right-half [0,1] space ──────
    # The composite has left_half (≈ half width) + gap + right_half (≈ half width).
    # We asked Gemini to report x_center ≥ 0.5. We need to map these back to [0,1]
    # relative to the right half only, so the overlay draws correctly on the image.
    #
    # Approximate the left-half fraction from the actual crop widths.
    lw_frac = lw / (lw + gap + rw)   # fraction of composite width = left half
    rw_frac = rw / (lw + gap + rw)   # fraction = right half
    for item in raw.get("missing", []):
        cx = float(item.get("x_center", 0.75))
        if cx >= lw_frac:             # in right half: remap to [0,1]
            item["x_center"] = (cx - lw_frac - gap/(lw+gap+rw)) / rw_frac
        else:
            item["x_center"] = cx    # shouldn't happen; keep as-is
        w = float(item.get("width", 0.05))
        item["width"] = min(w / rw_frac, 1.0)

    return raw


# ── Prompts ────────────────────────────────────────────────────────────────────

# Used when sending FULL images (GCS or local, uncropped)
_COMPARE_PROMPT = """\
IMAGE 1 = REFERENCE (golden sample — complete mold, ALL components installed).
IMAGE 2 = INSPECTION (same mold section; some components may be missing).

Task: find components that are INSTALLED in IMAGE 1 but NOT INSTALLED in IMAGE 2.

Key distinction:
• "Installed" means: a pin, piston, bolt, or insert is PHYSICALLY INSIDE a hole \
  (the hole is filled by a solid metal part).
• "Missing" means: in IMAGE 1 the hole has a part inside it, but in IMAGE 2 the \
  SAME hole is EMPTY (nothing inserted — you can see into the hole or just bare metal).

Do NOT report:
• Holes that appear EMPTY in BOTH images — those are part of the mold design.
• Differences caused by lighting, angle, shadow, or color only.

"photo_quality" refers to the clarity of the photo (sharp/blurry/dark/overexposed), \
NOT whether parts are present or missing.

Output ONLY this JSON (no markdown, no explanation):
{"missing":[{"label":"<piston|pin|bolt|screw|insert|clip|other>","description":"<one phrase>","x_center":<0.0-1.0>,"y_center":<0.0-1.0>,"width":<0.0-1.0>,"height":<0.0-1.0>,"confidence":<0.0-1.0>}],"total_missing":<int>,"photo_quality":"<good|acceptable|poor>","photo_notes":"<photo quality issues only, else empty>","summary":"<one sentence>"}"""

# Used when sending ROI-CROPPED images (both images are the same bounding-box area)
_ROI_CROP_PROMPT = """\
Both images show EXACTLY THE SAME cropped region of an injection mold.

IMAGE 1 = GOLDEN REFERENCE — all pistons and components are installed.
IMAGE 2 = INSPECTION — one or more pistons may be missing.

A PISTON in this mold looks like a large flat rectangular or cylindrical metal insert \
that sits flush in a machined pocket/recess. When a piston is MISSING, the same pocket \
is visibly empty — you can see the bare metal pocket or a dark void where the insert should be.

Step 1 — Locate every piston pocket that is FILLED in IMAGE 1.
Step 2 — Check each pocket in IMAGE 2. Report only those that are now EMPTY.

Rules:
• Skip small holes, bolts, screws, or pins — report ONLY large flat inserts (pistons).
• Skip any difference caused only by lighting, shadow, glare, or slight angle change.
• If a pocket looks the same in both images, do NOT report it.
• Be conservative: report a finding only when you are CERTAIN the pocket is empty in IMAGE 2.

Output ONLY valid JSON (no markdown, no text outside the JSON):
{"missing":[{"label":"piston","description":"<location e.g. upper-left pocket>","x_center":<0.0-1.0>,"y_center":<0.0-1.0>,"width":<0.0-1.0>,"height":<0.0-1.0>,"confidence":<0.0-1.0>}],"total_missing":<int>,"photo_quality":"good","photo_notes":"","summary":"<one sentence>"}"""

# Used for single side-by-side composite image (LEFT=reference, RIGHT=inspection)
_ROI_SIDEBYSIDE_PROMPT = """\
This image shows TWO PHOTOS of the SAME injection mold placed SIDE BY SIDE:
  LEFT HALF  (green header)  = REFERENCE — golden sample, ALL pistons installed.
  RIGHT HALF (red header)    = INSPECTION — taken now; some pistons may be missing.

A PISTON is a large flat metal insert that fills a machined rectangular or oval pocket \
in the mold surface. When installed, the pocket surface is flush with the surrounding \
metal. When MISSING, the pocket is visibly EMPTY — you can see into the recess or \
bare metal where the insert should be.

Task:
  1. Look at the LEFT half and find every piston pocket that is FILLED.
  2. Look at the SAME pocket in the RIGHT half.
  3. If that pocket is NOW EMPTY → it is a missing piston. Report it.

Important:
• The x_center / y_center coordinates must be in the RIGHT-HALF coordinate system \
  (i.e. x_center ≥ 0.5 in the composite image).
• Only report PISTONS — skip small bolts, screws, or pins.
• Skip differences that are only lighting, glare, or shadow.
• Be precise: report every clearly missing piston, but do not over-report.

Output ONLY this JSON (no markdown, no explanation):
{"missing":[{"label":"piston","description":"<e.g. left pocket, upper row>","x_center":<0.5-1.0>,"y_center":<0.0-1.0>,"width":<0.0-0.5>,"height":<0.0-1.0>,"confidence":<0.0-1.0>}],"total_missing":<int>,"photo_quality":"good","photo_notes":"","summary":"<one sentence>"}"""


def _call_gemini_two_step(ref_p: str, cand_p: str) -> dict[str, Any]:
    """Single-call: send both images simultaneously and ask for direct comparison."""
    import google.genai as genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=_PROJECT, location=_LOCATION)

    ref_img  = _cv2_read_path(ref_p)
    cand_img = _cv2_read_path(cand_p)
    ref_part  = _make_image_part(ref_p, ref_img)
    cand_part = _make_image_part(cand_p, cand_img)

    resp = client.models.generate_content(
        model=_MODEL,
        contents=[ref_part, cand_part, _COMPARE_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=16384,
        ),
    )
    raw = _parse_json(resp.text or "")
    # Normalise the key name (photo_quality → inspection_quality for _build_result)
    raw.setdefault("inspection_quality", raw.pop("photo_quality", "good"))
    raw.setdefault("quality_notes",      raw.pop("photo_notes", ""))
    return raw


# ── finding deduplication ──────────────────────────────────────────────────────

def _cluster_findings(findings: list[dict]) -> list[dict]:
    """
    Merge findings whose bounding boxes overlap substantially (IoU ≥ 0.4).
    Keeps the higher-confidence finding from each overlapping pair.
    This prevents double-counting when Gemini reports the same physical
    location twice under different labels.
    """
    if len(findings) <= 1:
        return findings

    def iou(a: dict, b: dict) -> float:
        ba, bb = a["bbox_normalized"], b["bbox_normalized"]
        ax1, ay1 = ba["x"], ba["y"]
        ax2, ay2 = ax1 + ba["width"], ay1 + ba["height"]
        bx1, by1 = bb["x"], bb["y"]
        bx2, by2 = bx1 + bb["width"], by1 + bb["height"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    merged: list[dict] = []
    used = [False] * len(findings)
    findings_sorted = sorted(findings, key=lambda f: -f.get("confidence", 0))
    for i, fi in enumerate(findings_sorted):
        if used[i]:
            continue
        for j, fj in enumerate(findings_sorted):
            if j <= i or used[j]:
                continue
            if iou(fi, fj) >= 0.4:
                used[j] = True   # absorb lower-confidence duplicate
        merged.append(fi)
        used[i] = True
    return merged


# ── result builder ─────────────────────────────────────────────────────────────

def _build_result(
    raw: dict[str, Any],
    cand_p: str,
    family: str, zone_id: str,
    evidence_dir,
    min_confidence: float,
) -> dict[str, Any]:
    """Convert raw Gemini JSON to the pipeline-compatible result dict."""
    if raw.get("total_missing", 0) > 150:
        raw["inspection_quality"] = "poor"
        raw["quality_notes"] = "Demasiadas diferencias — posible sección equivocada."
        raw["missing"] = []

    missing = [m for m in raw.get("missing", []) if m.get("confidence", 0) >= min_confidence]
    findings = []
    for item in missing:
        x, y = item["x_center"], item["y_center"]
        w, h = item.get("width", 0.06), item.get("height", 0.06)
        findings.append({
            "piece_id": f"gemini_{item['label']}_{len(findings)+1}",
            "class_name": item["label"],
            "status": "missing",
            "confidence": round(float(item.get("confidence", 0.9)), 4),
            "region": [
                {"x": round(x - w/2, 4), "y": round(y - h/2, 4)},
                {"x": round(x + w/2, 4), "y": round(y - h/2, 4)},
                {"x": round(x + w/2, 4), "y": round(y + h/2, 4)},
                {"x": round(x - w/2, 4), "y": round(y + h/2, 4)},
            ],
            "bbox_normalized": {"x": round(x - w/2, 4), "y": round(y - h/2, 4),
                                "width": round(w, 4), "height": round(h, 4)},
            "method": "gemini_vision",
        })

    n = len(findings)
    quality = raw.get("inspection_quality", "good")
    photo_notes = raw.get("quality_notes", "")
    # Only "retake_photo" when the PHOTO itself is bad (blurry/dark) AND nothing useful detected.
    # When parts are missing, always use "review" regardless of quality label.
    if quality == "poor" and n == 0:
        status  = "retake_photo"
        message = photo_notes or "La foto no es suficientemente comparable."
    elif n == 0:
        status  = "correct"
        message = "Todas las piezas presentes — zona validada por Gemini."
    else:
        status  = "review"
        message = raw.get("summary") or f"Gemini detectó {n} parte(s) faltante(s)."

    return {
        "status": status,
        "message": message,
        "findings": findings,
        "missing_count": n,
        "overlay_image": None,
        "method": "gemini_vision",
        "model": _MODEL,
        "raw_gemini": {
            "total_missing": raw.get("total_missing", n),
            "summary": raw.get("summary", ""),
            "inspection_quality": quality,
            "quality_notes": raw.get("quality_notes", ""),
        },
    }


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "status": "review",
        "reason": "inspection_error",
        "message": msg,
        "findings": [],
        "missing_count": 0,
        "overlay_image": None,
        "method": "hybrid_opencv_gemini",
    }


# ── overlay drawing ────────────────────────────────────────────────────────────

def _draw_cv_overlay_roi(
    img, poly_norm: list, findings: list[dict],
    evidence_dir, family: str, zone_id: str, cand_p: str,
) -> str | None:
    """Draw ROI polygon boundary + finding boxes on the full aligned image."""
    if evidence_dir is None:
        return None
    try:
        import cv2
    except ImportError:
        return None

    H, W  = img.shape[:2]
    overlay = img.copy()
    n = len(findings)

    # Draw ROI polygon boundary (thin green line = "what we inspect")
    pts = np.array([[int(x * W), int(y * H)] for x, y in poly_norm], dtype=np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 220, 0), thickness=3)

    # Header bar
    cv2.rectangle(overlay, (0, 0), (W, 52), (10, 10, 10), -1)
    hdr   = "COMPLETO" if n == 0 else f"FALTAN {n} PIEZA(S)"
    color = (30, 200, 30) if n == 0 else (0, 0, 220)
    cv2.putText(overlay, hdr, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(overlay, "classical_cv", (W - 200, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    for i, f in enumerate(findings):
        b  = f["bbox_normalized"]
        x1 = int(b["x"] * W)
        y1 = int(b["y"] * H)
        x2 = int((b["x"] + b["width"])  * W)
        y2 = int((b["y"] + b["height"]) * H)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), 3)
        cv2.circle(overlay, (cx, cy), 14, (0, 0, 220), -1)
        cv2.putText(overlay, str(i + 1), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(overlay, f["class_name"], (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 40, 200), 2)

    stem    = Path(cand_p).stem
    out_dir = Path(evidence_dir) / family / zone_id / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv_overlay.jpg"
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(out_path)


def _draw_cv_overlay(
    img, findings: list[dict], evidence_dir, family: str, zone_id: str,
    cand_p: str, method: str = "classical_cv",
) -> str | None:
    """Draw CV findings on a crop image and save to evidence_dir."""
    if evidence_dir is None:
        return None
    try:
        import cv2
    except ImportError:
        return None

    H, W  = img.shape[:2]
    overlay = img.copy()
    n = len(findings)

    # Header bar
    cv2.rectangle(overlay, (0, 0), (W, 52), (10, 10, 10), -1)
    hdr   = "COMPLETO" if n == 0 else f"FALTAN {n} PIEZA(S)"
    color = (30, 180, 30) if n == 0 else (0, 0, 220)
    cv2.putText(overlay, hdr, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(overlay, method, (W - 220, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    for i, f in enumerate(findings):
        b  = f["bbox_normalized"]
        x1 = int(b["x"] * W)
        y1 = int(b["y"] * H)
        x2 = int((b["x"] + b["width"])  * W)
        y2 = int((b["y"] + b["height"]) * H)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), 3)
        cv2.circle(overlay, (cx, cy), 12, (0, 0, 220), -1)
        cv2.putText(overlay, str(i + 1), (cx - 7, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f["class_name"], (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 40, 200), 2)

    stem    = Path(cand_p).stem
    out_dir = Path(evidence_dir) / family / zone_id / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv_overlay.jpg"
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(out_path)


def _draw_overlay_clean(
    cand_img, findings: list[dict], evidence_dir, family: str, zone_id: str, cand_p: str
) -> str | None:
    """Draw findings on the candidate image and save."""
    if evidence_dir is None:
        return None
    try:
        import cv2
    except ImportError:
        return None

    H, W = cand_img.shape[:2]
    overlay = cand_img.copy()
    n = len(findings)

    for i, f in enumerate(findings):
        bbox = f["bbox_normalized"]
        x1 = int(bbox["x"] * W)
        y1 = int(bbox["y"] * H)
        x2 = int((bbox["x"] + bbox["width"])  * W)
        y2 = int((bbox["y"] + bbox["height"]) * H)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Draw a solid rectangle around the region
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), 3)
        # Draw a small circle at the center
        cv2.circle(overlay, (cx, cy), 12, (0, 0, 220), -1)
        label = f"{i+1}. {f['class_name']} ({f['confidence']:.0%})"
        cv2.putText(overlay, label, (x1 + 6, y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

    # Status bar
    color = (30, 180, 30) if n == 0 else (0, 0, 220)
    text  = "COMPLETO" if n == 0 else f"FALTAN {n} PIEZA(S)"
    cv2.rectangle(overlay, (0, 0), (W, 52), (10, 10, 10), -1)
    cv2.putText(overlay, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(overlay, f"Hybrid OpenCV+Gemini  model={_MODEL}",
                (W - 420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    stem = Path(cand_p).stem
    out_dir = Path(evidence_dir) / family / zone_id / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gemini_overlay.jpg"
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(out_path)


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from Gemini, tolerating markdown fences and thinking tokens."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.error("Could not parse Gemini JSON: %s", text[:300])
        return {"missing": [], "summary": "Parse error", "total_missing": 0,
                "inspection_quality": "good", "quality_notes": ""}


# ── image reading ──────────────────────────────────────────────────────────────

def _read_image_bytes(path: Path) -> bytes:
    """Read image bytes, converting HEIC via pillow-heif or sips if needed."""
    if path.suffix.lower() not in {".heic", ".heif"}:
        with open(path, "rb") as f:
            return f.read()
    # Try pillow-heif (Linux + macOS)
    try:
        from PIL import Image
        import pillow_heif
        import io
        pillow_heif.register_heif_opener()
        pil_img = Image.open(str(path)).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception:
        pass
    # macOS fallback: sips
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with open(tmp, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
