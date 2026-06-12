#!/usr/bin/env python3
"""Bootstrap benchmark annotations from classified present/missing photo pairs.

Consecutive present/missing shots share pose almost exactly, so an aligned
diff localizes the removed pieces reliably (unlike cross-pose inspection,
where diffing is invalid). Per section this emits:

  benchmarks/annotations/{dataset}__{vista}__{section}.json
    golden_images, eval_images (with missing_piece_ids), zone_polygon_norm,
    pieces (bbox_norm, control flag)
  benchmarks/annotations/overlays/{...}.jpg   — visual verification

Controls are unchanged-region boxes labeled present in every image; they give
the benchmark present-class coverage at varied locations.

Usage:
  python scripts/presence_annotate_bootstrap.py \
      --dataset try-photos-mold-a \
      --dataset "try-photos-last-mold-with-without pieces" \
      --out benchmarks/annotations
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mold_inspection.presence.alignment import align_zone, warp_to_golden  # noqa: E402
from mold_inspection.presence.imageio import clahe_gray, read_image  # noqa: E402

WORK_SIDE = 2000
MIN_BLOB_FRAC = 0.0001
MAX_BLOB_FRAC = 0.05
MAX_BLOBS_PER_PAIR = 8
N_CONTROLS = 5


def img_number(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def pair_diff_boxes(present_bgr: np.ndarray, missing_bgr: np.ndarray) -> list[tuple[float, float, float, float]] | None:
    """Aligned diff between a consecutive pair; returns normalized boxes in the PRESENT frame."""
    alignment = align_zone(present_bgr, missing_bgr, None, max_side=1600)
    if not alignment.ok:
        return None
    warped = warp_to_golden(missing_bgr, alignment, present_bgr.shape)

    valid = warp_to_golden(np.full(missing_bgr.shape[:2], 255, np.uint8)[..., None].repeat(3, 2),
                           alignment, present_bgr.shape)
    valid = cv2.erode(cv2.cvtColor(valid, cv2.COLOR_BGR2GRAY), np.ones((9, 9), np.uint8))

    g1 = clahe_gray(cv2.cvtColor(present_bgr, cv2.COLOR_BGR2GRAY))
    g2 = clahe_gray(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
    diff = cv2.absdiff(g1, g2)
    diff = cv2.GaussianBlur(diff, (0, 0), 2.0)
    diff[valid == 0] = 0

    vals = diff[valid > 0]
    if vals.size == 0:
        return None
    thr = max(float(np.percentile(vals, 99.3)), 30.0)
    hot = (diff >= thr).astype(np.uint8) * 255
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    h, w = diff.shape
    area_img = h * w
    contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_BLOBS_PER_PAIR * 2]:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_FRAC * area_img <= area <= MAX_BLOB_FRAC * area_img):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if area / max(bw * bh, 1) < 0.20:
            continue
        boxes.append((x / w, y / h, (x + bw) / w, (y + bh) / h))
        if len(boxes) >= MAX_BLOBS_PER_PAIR:
            break
    return boxes


def transform_boxes(boxes, h_mat, src_shape, dst_shape):
    """Map normalized boxes through a pixel-space homography."""
    sh, sw = src_shape[:2]
    dh, dw = dst_shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        corners = np.float32([[x1 * sw, y1 * sh], [x2 * sw, y1 * sh],
                              [x2 * sw, y2 * sh], [x1 * sw, y2 * sh]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, h_mat).reshape(-1, 2)
        out.append((float(proj[:, 0].min() / dw), float(proj[:, 1].min() / dh),
                    float(proj[:, 0].max() / dw), float(proj[:, 1].max() / dh)))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def center_dist(a, b) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot(ax - bx, ay - by)


def cluster_boxes(tagged_boxes: list[tuple[int, tuple]], n_pairs: int) -> list[dict]:
    """Greedy IoU/center clustering; keep clusters seen in enough pairs."""
    clusters: list[dict] = []
    for pair_idx, box in tagged_boxes:
        size = math.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1e-9))
        placed = False
        for cl in clusters:
            ref = cl["box"]
            if iou(box, ref) >= 0.25 or center_dist(box, ref) < 0.6 * size:
                cl["members"].append((pair_idx, box))
                arr = np.array([b for _, b in cl["members"]])
                cl["box"] = tuple(np.median(arr, axis=0).tolist())
                placed = True
                break
        if not placed:
            clusters.append({"box": box, "members": [(pair_idx, box)]})

    min_pairs = 1 if n_pairs == 1 else max(2, math.ceil(0.4 * n_pairs))
    kept = []
    for cl in clusters:
        pairs_seen = {idx for idx, _ in cl["members"]}
        if len(pairs_seen) >= min_pairs:
            cl["pairs_seen"] = sorted(pairs_seen)
            kept.append(cl)
    return kept


def point_in_zone(x: float, y: float, zone: list[tuple[float, float]]) -> bool:
    poly = np.array(zone, np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def box_in_zone(box: tuple, zone: list[tuple[float, float]]) -> bool:
    """Box counts as inside when its center and at least 3 of 4 corners are inside."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if not point_in_zone(cx, cy, zone):
        return False
    corners_in = sum(point_in_zone(px, py, zone)
                     for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return corners_in >= 3


def make_zone_polygon(piece_boxes: list[tuple]) -> list[tuple[float, float]]:
    arr = np.array(piece_boxes)
    x1, y1 = arr[:, 0].min(), arr[:, 1].min()
    x2, y2 = arr[:, 2].max(), arr[:, 3].max()
    mx, my = max(0.12, (x2 - x1) * 0.35), max(0.12, (y2 - y1) * 0.35)
    x1, y1 = max(0.02, x1 - mx), max(0.02, y1 - my)
    x2, y2 = min(0.98, x2 + mx), min(0.98, y2 + my)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def make_controls(piece_boxes: list[tuple], zone: list[tuple[float, float]], n: int = N_CONTROLS) -> list[tuple]:
    if not piece_boxes:
        return []
    arr = np.array(piece_boxes)
    mw = float(np.median(arr[:, 2] - arr[:, 0]))
    mh = float(np.median(arr[:, 3] - arr[:, 1]))
    zarr = np.array(zone)
    zx1, zy1 = float(zarr[:, 0].min()), float(zarr[:, 1].min())
    zx2, zy2 = float(zarr[:, 0].max()), float(zarr[:, 1].max())
    controls = []
    rng = np.random.default_rng(11)
    for _ in range(600):
        if len(controls) >= n:
            break
        cx = rng.uniform(zx1 + mw, max(zx2 - mw, zx1 + mw + 1e-6))
        cy = rng.uniform(zy1 + mh, max(zy2 - mh, zy1 + mh + 1e-6))
        cand = (cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2)
        if not box_in_zone(cand, zone):
            continue
        clear = all(iou(cand, p) < 0.02 and center_dist(cand, p) > 0.8 * max(mw, mh) for p in piece_boxes)
        clear = clear and all(iou(cand, c) < 0.02 for c in controls)
        if clear:
            controls.append(cand)
    return controls


def process_section(section_dir: Path, out_dir: Path, dataset_name: str,
                    zones: dict[str, list] | None = None) -> dict | None:
    present_dir, missing_dir = section_dir / "present", section_dir / "missing"
    if not present_dir.is_dir() or not missing_dir.is_dir():
        return None
    present = sorted(present_dir.glob("*.jpg"), key=img_number) + sorted(present_dir.glob("*.JPG"), key=img_number)
    missing = sorted(missing_dir.glob("*.jpg"), key=img_number) + sorted(missing_dir.glob("*.JPG"), key=img_number)
    if len(present) < 2 or len(missing) < 1:
        print(f"  [skip] {section_dir.name}: present={len(present)} missing={len(missing)}")
        return None

    vista = section_dir.parent.name
    section = section_dir.name
    slug = f"{dataset_name.replace(' ', '_')}__{vista}__{section}"
    print(f"  [{slug}] present={len(present)} missing={len(missing)}")

    golden_path = present[0]
    golden = read_image(golden_path, max_side=WORK_SIDE)

    cache: dict[Path, np.ndarray] = {golden_path: golden}

    def load(p: Path) -> np.ndarray:
        if p not in cache:
            cache[p] = read_image(p, max_side=WORK_SIDE)
        return cache[p]

    # Pair each missing image with the nearest-numbered present image.
    tagged_boxes: list[tuple[int, tuple]] = []
    pair_records = []
    for pair_idx, mpath in enumerate(missing):
        mnum = img_number(mpath)
        ppath = min(present, key=lambda p: abs(img_number(p) - mnum))
        pimg, mimg = load(ppath), load(mpath)
        boxes = pair_diff_boxes(pimg, mimg)
        if boxes is None:
            print(f"    pair {mpath.name}<->{ppath.name}: alignment FAILED")
            pair_records.append({"missing": str(mpath), "present": str(ppath), "ok": False})
            continue
        # Map boxes from the paired-present frame into the golden frame.
        if ppath != golden_path:
            to_golden = align_zone(golden, pimg, None, max_side=1600)
            if not to_golden.ok:
                print(f"    pair {mpath.name}: present->golden alignment FAILED")
                pair_records.append({"missing": str(mpath), "present": str(ppath), "ok": False})
                continue
            boxes = transform_boxes(boxes, to_golden.homography, pimg.shape, golden.shape)
        tagged_boxes.extend((pair_idx, b) for b in boxes)
        pair_records.append({"missing": str(mpath), "present": str(ppath), "ok": True, "blobs": len(boxes)})

    ok_pairs = [r for r in pair_records if r.get("ok")]
    if not tagged_boxes or not ok_pairs:
        print(f"    no usable pairs -> skipped")
        return None

    # Manual mold-body zone: discard diff blobs outside it (floor, mats, people,
    # neighboring molds) BEFORE clustering so they never become pieces.
    manual_zone = (zones or {}).get(slug)
    if manual_zone:
        manual_zone = [tuple(p) for p in manual_zone]
        n_before = len(tagged_boxes)
        tagged_boxes = [(idx, b) for idx, b in tagged_boxes if box_in_zone(b, manual_zone)]
        if n_before != len(tagged_boxes):
            print(f"    zone filter: {n_before} -> {len(tagged_boxes)} blobs")
        if not tagged_boxes:
            print(f"    all blobs outside manual zone -> skipped")
            return None

    clusters = cluster_boxes(tagged_boxes, n_pairs=len(ok_pairs))
    if not clusters:
        print(f"    no stable clusters -> skipped")
        return None

    piece_boxes = [cl["box"] for cl in clusters]
    zone = manual_zone if manual_zone else make_zone_polygon(piece_boxes)
    controls = make_controls(piece_boxes, zone)

    pieces = []
    for i, cl in enumerate(clusters):
        pieces.append({"id": f"piece_{i+1:02d}", "bbox_norm": list(cl["box"]),
                       "control": False, "pairs_seen": cl["pairs_seen"]})
    for j, box in enumerate(controls):
        pieces.append({"id": f"control_{j+1:02d}", "bbox_norm": list(box), "control": True})

    # Per missing image: which clusters its pair contributed to.
    eval_images = []
    golden_count = min(3, max(2, len(present) // 3))
    golden_paths = present[:golden_count]
    for p in present[golden_count:]:
        eval_images.append({"path": str(p), "missing_piece_ids": []})
    for pair_idx, mpath in enumerate(missing):
        rec = pair_records[pair_idx]
        if not rec.get("ok"):
            continue
        ids = [f"piece_{i+1:02d}" for i, cl in enumerate(clusters) if pair_idx in cl["pairs_seen"]]
        eval_images.append({"path": str(mpath), "missing_piece_ids": ids})

    record = {
        "slug": slug,
        "dataset": dataset_name,
        "vista": vista,
        "section": section,
        "golden_images": [str(p) for p in golden_paths],
        "eval_images": eval_images,
        "zone_polygon_norm": [list(p) for p in zone],
        "pieces": pieces,
        "pairs": pair_records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(record, indent=2))

    # Verification overlay on the golden image.
    overlay = golden.copy()
    gh, gw = overlay.shape[:2]
    zone_px = np.array([[int(x * gw), int(y * gh)] for x, y in zone], np.int32)
    cv2.polylines(overlay, [zone_px], True, (255, 160, 0), 3)
    for piece in pieces:
        x1, y1, x2, y2 = piece["bbox_norm"]
        color = (0, 200, 0) if piece["control"] else (0, 0, 230)
        cv2.rectangle(overlay, (int(x1 * gw), int(y1 * gh)), (int(x2 * gw), int(y2 * gh)), color, 3)
        cv2.putText(overlay, piece["id"].split("_")[-1], (int(x1 * gw), max(20, int(y1 * gh) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    ov_dir = out_dir / "overlays"
    ov_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(ov_dir / f"{slug}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

    n_real = sum(1 for p in pieces if not p["control"])
    print(f"    -> {n_real} pieces + {len(controls)} controls, {len(eval_images)} eval images")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    help="dataset folder name under the repo root (expects <name>/classified/...)")
    ap.add_argument("--out", default="benchmarks/annotations")
    ap.add_argument("--zones", default="benchmarks/zones.json",
                    help="JSON of slug -> mold-body polygon (normalized); blobs outside are discarded")
    args = ap.parse_args()

    zones: dict[str, list] = {}
    zones_path = REPO_ROOT / args.zones
    if zones_path.is_file():
        zones = {k: v for k, v in json.loads(zones_path.read_text()).items()
                 if not k.startswith("_")}
        print(f"Loaded {len(zones)} manual zones from {zones_path}")

    out_dir = REPO_ROOT / args.out
    total = 0
    for ds in args.dataset:
        classified = REPO_ROOT / ds / "classified"
        if not classified.is_dir():
            print(f"[warn] {classified} not found")
            continue
        print(f"== {ds} ==")
        for vista_dir in sorted(p for p in classified.iterdir() if p.is_dir()):
            for section_dir in sorted(p for p in vista_dir.iterdir() if p.is_dir()):
                if process_section(section_dir, out_dir, ds, zones=zones):
                    total += 1
    print(f"\nDone: {total} section annotation files in {out_dir}")


if __name__ == "__main__":
    main()
