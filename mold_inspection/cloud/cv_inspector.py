"""
Deterministic classical-CV mold inspection — no LLM, no network calls.

Compares an inspection photo against one or more golden references entirely with
OpenCV: ORB homography alignment, per-tile brightness normalisation, SSIM gate,
adaptive in-ROI pixel diff, and contour classification. Fully reproducible and
biased toward false-rejection (a localized structural change is never silently
approved by a diluted global mean).

Two entry points:
  • ``inspect_with_cv``            — single reference vs candidate.
  • ``inspect_with_cv_consensus``  — candidate vs every reference, keeping only
    findings corroborated by a majority of references (kills lighting/pose
    artifacts that show up against just one reference).

Usage:
    result = inspect_with_cv_consensus(
        reference_image_paths=["gs://... or /local/ref1", "/local/ref2"],
        candidate_image_path="gs://... or /local/path",
        family="my_family",
        zone_id="zona_01",
        evidence_dir="/tmp/evidence",
    )
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MAX_W        = 1400     # resize long side to this (preserves memory & compute)
_TILE         = 120      # tile size for SSIM-map brightness normalisation
_SSIM_THR     = 0.55     # tiles below this are "structurally different"
_SAME_IMG_THR = 0.05     # global mean SSIM diff below this → images near-identical

_CV_TILE       = 80      # tile size for adaptive-diff brightness normalisation
_CV_SIGMA      = 2.5     # threshold = mean + CV_SIGMA * std (adaptive)
_CV_PISTON_THR = 0.012   # blob area fraction of ROI ≥ this → piston; below → bolt

# ROI polygon — normalized (x, y) in [0,1] relative to the loaded image dimensions.
# Applied to every image via the ORB warp; excludes outer frame, floor, background.
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
    roi_polygon: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """
    Pure classical CV mold inspection — no LLM, fully deterministic.

    Pipeline:
      1. Load both images (HEIC + GCS handled transparently).
      2. Scale to _MAX_W. ORB-align candidate to reference.
      3. Build ROI mask from the polygon (in reference-image space).
      4. SSIM gate: short-circuit to "correct" ONLY when the global mean diff
         < _SAME_IMG_THR AND there is no localized structural cluster inside the
         ROI. A localized cluster falls through to the diff pipeline so a small
         missing piece is never approved by a diluted global mean.
      5. Per-tile brightness normalise inside the ROI to cancel lighting.
      6. Adaptive-threshold pixel diff inside ROI: mean + _CV_SIGMA × std.
      7. Morphological close+open to merge blobs and remove micro-noise.
      8. Contour size classification (relative to ROI area):
           blob_area / roi_area ≥ _CV_PISTON_THR  → piston ; below → bolt.
      9. Draw overlay on the aligned candidate, with ROI boundary shown.

    Returns a dict compatible with the pipeline piece_inspection format. On a load
    or alignment failure it returns ``_error_result`` (reason "inspection_error"),
    which the consensus orchestrator treats as an unusable reference.
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
    roi_mask   = _make_roi_mask(IH, IW, poly)
    active     = cv2.bitwise_and(valid_orb, roi_mask)   # inside polygon AND valid warp
    roi_area   = int(np.count_nonzero(active))
    if roi_area < 1000:
        return _error_result("ROI mask is empty after ORB alignment — cannot compare.")

    # ── 4. SSIM gate ─────────────────────────────────────────────────────────────
    # Approve as "correct" ONLY when global mean SSIM diff < _SAME_IMG_THR AND no
    # localized structural cluster inside the ROI. The localized guard is essential:
    # a single missing piece moves the global mean by well under _SAME_IMG_THR, so
    # the mean alone would silently approve a faulty mold.
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


def inspect_with_cv_consensus(
    reference_image_paths: list[str | Path],
    candidate_image_path: str | Path,
    family: str,
    zone_id: str,
    evidence_dir: str | Path | None = None,
    roi_polygon: list[tuple[float, float]] | None = None,
    min_support: int | None = None,
) -> dict[str, Any] | None:
    """
    Run the deterministic CV inspection against EVERY golden reference and keep
    only findings corroborated by at least ``min_support`` of the references that
    produced a usable comparison.

    Rationale: a genuinely missing piece differs from *all* golden references, so
    it earns full support; a lighting/pose artifact differs from at most one
    reference and is suppressed. With a single reference this degrades exactly to
    ``inspect_with_cv``.

    The references are golden shots of the same zone, framed similarly, so
    findings (normalized to image [0,1]) are comparable across references; a
    lenient IoU (≥0.3) clusters the same physical location.

    ``min_support`` defaults to a strict majority (``len(used)//2 + 1``).

    Returns a pipeline-compatible piece_inspection dict, or ``None`` when no
    reference could be compared (so the caller can fall back to the pixel diff).
    """
    refs = [r for r in reference_image_paths if r]
    if not refs:
        return None

    runs: list[dict[str, Any]] = []
    for ref_path in refs:
        try:
            result = inspect_with_cv(
                reference_image_path=ref_path,
                candidate_image_path=candidate_image_path,
                family=family,
                zone_id=zone_id,
                evidence_dir=evidence_dir,
                roi_polygon=roi_polygon,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("CV run failed for reference %s: %s", ref_path, exc)
            continue
        if result.get("reason") == "inspection_error":
            logger.info("Skipping unusable reference %s: %s", ref_path, result.get("message"))
            continue
        runs.append(result)

    if not runs:
        return None

    n_refs   = len(runs)
    required = min_support if min_support is not None else (n_refs // 2 + 1)
    required = max(1, min(required, n_refs))

    # Cluster findings across runs by IoU; a cluster's support = distinct refs.
    all_findings = [
        (idx, f)
        for idx, run in enumerate(runs)
        for f in run.get("findings", [])
        if isinstance(f.get("bbox_normalized"), dict)
    ]
    all_findings.sort(key=lambda pair: pair[1].get("confidence", 0.0), reverse=True)

    clusters: list[dict[str, Any]] = []
    for idx, finding in all_findings:
        placed = False
        for cluster in clusters:
            if any(_iou(finding, member) >= 0.3 for member in cluster["members"]):
                cluster["members"].append(finding)
                cluster["refs"].add(idx)
                placed = True
                break
        if not placed:
            clusters.append({"members": [finding], "refs": {idx}})

    kept: list[dict[str, Any]] = []
    for cluster in clusters:
        support = len(cluster["refs"])
        if support < required:
            continue
        rep = max(cluster["members"], key=lambda f: f.get("confidence", 0.0))
        kept.append({
            **rep,
            "support": support,
            "support_ratio": round(support / n_refs, 3),
            "method": "classical_cv_consensus",
        })

    n = len(kept)
    status = "review" if n else "correct"
    message = (
        "Todas las piezas presentes — confirmado contra todas las referencias."
        if n == 0
        else f"Detectadas {n} pieza(s) faltante(s), confirmada(s) por ≥{required} de {n_refs} referencias."
    )

    overlay = _consensus_overlay(candidate_image_path, kept, evidence_dir, family, zone_id, roi_polygon)
    if overlay is None:
        # Fall back to a run's own overlay so the UI still has evidence.
        overlay = next((r.get("overlay_image") for r in runs if r.get("overlay_image")), None)

    mean_diffs = [r.get("cv_stats", {}).get("mean_diff") for r in runs]
    mean_diffs = [d for d in mean_diffs if isinstance(d, (int, float))]
    return {
        "status":        status,
        "message":       message,
        "findings":      kept,
        "missing_count": n,
        "overlay_image": overlay,
        "method":        "classical_cv_consensus",
        "cv_stats": {
            "references_total": len(refs),
            "references_used":  n_refs,
            "required_support": required,
            "min_mean_diff":    round(min(mean_diffs), 4) if mean_diffs else None,
            "max_mean_diff":    round(max(mean_diffs), 4) if mean_diffs else None,
        },
    }


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two findings' normalized bounding boxes."""
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


def _consensus_overlay(
    candidate_image_path: str | Path,
    findings: list[dict[str, Any]],
    evidence_dir, family: str, zone_id: str,
    roi_polygon: list[tuple[float, float]] | None,
) -> str | None:
    """Redraw the consensus findings on the candidate image (normalized space)."""
    if evidence_dir is None:
        return None
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None
    cand_img = _cv2_read_path(str(candidate_image_path))
    if cand_img is None:
        return None
    cand_img = _scale(cand_img, _MAX_W)
    poly = roi_polygon if roi_polygon is not None else _DEFAULT_ROI_POLY_NORM
    return _draw_cv_overlay_roi(
        cand_img, poly, findings, evidence_dir, family, zone_id, str(candidate_image_path)
    )


# ── image loading ──────────────────────────────────────────────────────────────

def _cv2_read_path(path: str):
    """Load image from local file or GCS URI (uses google-cloud-storage, no gsutil)."""
    try:
        import cv2  # noqa: F401
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
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # macOS fallback: sips
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(p), "--out", tmp],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return cv2.imread(tmp, cv2.IMREAD_COLOR)
    except Exception:
        return None
    finally:
        if tmp and os.path.exists(tmp):
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

    win = min(11, min(H, W) // 4 * 2 + 1)  # odd
    _, ssim_map = ssim(ref_u8, cand_u8, win_size=win, full=True, data_range=255)
    return np.clip(1.0 - ssim_map, 0, 1).astype(np.float32)


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


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "status": "review",
        "reason": "inspection_error",
        "message": msg,
        "findings": [],
        "missing_count": 0,
        "overlay_image": None,
        "method": "classical_cv",
    }
