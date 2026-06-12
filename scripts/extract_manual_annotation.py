#!/usr/bin/env python3
"""Extract a hand-drawn annotation (green zone + red part outlines) by diffing
an annotated photo against its clean original.

Output: a registration JSON (zone polygon + per-part bboxes, normalized) and a
verification overlay rendered from the extracted data.

Usage:
  python scripts/extract_manual_annotation.py \
      --annotated "anotated-example/IMG_2440 annotated.jpg" \
      --clean "anotated-example/IMG_2440.jpg" \
      --out anotated-example/IMG_2440_registration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DIFF_THR = 35          # pixel counts as drawn when any channel differs this much
COLOR_MARGIN = 30      # dominant-channel margin to call a stroke red vs green
MIN_PART_AREA = 60     # px^2 at working scale, drops dust
WORK_SIDE = 2200


def load_pair(annotated_path: str, clean_path: str) -> tuple[np.ndarray, np.ndarray]:
    ann = cv2.imread(annotated_path)
    cln = cv2.imread(clean_path)
    if ann is None or cln is None:
        raise SystemExit("could not read input images")
    if ann.shape[:2] != cln.shape[:2]:
        cln = cv2.resize(cln, (ann.shape[1], ann.shape[0]))
    s = WORK_SIDE / max(ann.shape[:2])
    if s < 1.0:
        size = (int(ann.shape[1] * s), int(ann.shape[0] * s))
        ann, cln = cv2.resize(ann, size), cv2.resize(cln, size)
    return ann, cln


def stroke_masks(ann: np.ndarray, cln: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = cv2.absdiff(ann, cln).max(axis=2)
    drawn = diff > DIFF_THR
    b, g, r = ann[..., 0].astype(int), ann[..., 1].astype(int), ann[..., 2].astype(int)
    green = drawn & (g > r + COLOR_MARGIN) & (g > b + COLOR_MARGIN)
    red = drawn & (r > g + COLOR_MARGIN) & (r > b + COLOR_MARGIN)
    return green.astype(np.uint8) * 255, red.astype(np.uint8) * 255


def zone_from_green(green: np.ndarray) -> list[list[float]]:
    h, w = green.shape
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SystemExit("no green zone stroke found")
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.0028 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    return [[float(x / w), float(y / h)] for x, y in poly]


def parts_from_red(red: np.ndarray) -> list[dict]:
    h, w = red.shape
    # Bridge small gaps in thin outlines so each part forms one component.
    fat = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours, _ = cv2.findContours(fat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < MIN_PART_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        parts.append({
            "bbox_norm": [x / w, y / h, (x + bw) / w, (y + bh) / h],
            "area_px": float(cv2.contourArea(cnt)),
        })
    # Stable ids ordered top-to-bottom then left-to-right.
    parts.sort(key=lambda p: (round(p["bbox_norm"][1], 2), p["bbox_norm"][0]))
    for i, p in enumerate(parts):
        p["id"] = f"part_{i+1:03d}"
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ann, cln = load_pair(args.annotated, args.clean)
    green, red = stroke_masks(ann, cln)
    zone = zone_from_green(green)
    parts = parts_from_red(red)

    record = {
        "source_image": str(Path(args.clean).resolve()),
        "annotated_image": str(Path(args.annotated).resolve()),
        "zone_polygon_norm": zone,
        "parts": parts,
        "n_parts": len(parts),
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=2))
    print(f"zone vertices: {len(zone)}   parts: {len(parts)}")

    # Verification overlay rendered purely from the EXTRACTED data.
    vis = cln.copy()
    h, w = vis.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in zone], np.int32)
    cv2.polylines(vis, [pts], True, (0, 230, 0), 4)
    for p in parts:
        x1, y1, x2, y2 = p["bbox_norm"]
        cv2.rectangle(vis, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 235), 2)
    vis_path = out.with_suffix(".check.jpg")
    cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out} and {vis_path}")


if __name__ == "__main__":
    main()
