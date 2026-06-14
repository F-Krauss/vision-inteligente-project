#!/usr/bin/env python3
"""Extract a hand-drawn annotation (green zone + red part outlines) by diffing
an annotated photo against its clean original.

The operator draws, on top of the clean photo, a single green loop for the
inspection zone and one red loop around every part to map. This script recovers
each of those as structured data.

Key idea — parts are *closed outlines*, not blobs. Instead of fattening the red
strokes and boxing the result (which merges neighbours and fragments broken
loops), we seal small gaps in the strokes, then flood-fill the background so each
loop's enclosed interior becomes one solid region. One region == one part, which
tracks the hand drawing far more faithfully.

Output: a registration JSON (zone polygon + per-part bbox **and** polygon,
normalized) plus a verification overlay rendered purely from the extracted data.

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

DIFF_THR = 30           # a pixel counts as "drawn" when any channel differs this much
COLOR_MARGIN = 24       # dominant-channel margin to call a stroke red vs green
MIN_AREA_FRAC = 2.2e-5  # drop parts smaller than this fraction of the image (dust)
MAX_AREA_FRAC = 0.10    # drop regions larger than this (stray fills / zone leakage)
MIN_FILL_RATIO = 0.10   # region must fill at least this share of its bbox (drop slivers)


def load_pair(annotated_path: str, clean_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the annotated photo and its clean twin at full resolution, sized to
    match. Full res preserves the thin hand strokes that downscaling would break."""
    ann = cv2.imread(annotated_path)
    cln = cv2.imread(clean_path)
    if ann is None or cln is None:
        raise SystemExit("could not read input images")
    if ann.shape[:2] != cln.shape[:2]:
        cln = cv2.resize(cln, (ann.shape[1], ann.shape[0]))
    return ann, cln


def _odd(value: float, floor: int = 3) -> int:
    k = max(floor, int(round(value)))
    return k + 1 if k % 2 == 0 else k


def stroke_masks(ann: np.ndarray, cln: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isolate the green (zone) and red (parts) strokes the operator added.

    Combines a difference gate (the stroke changed the pixel vs the clean photo)
    with an HSV colour gate (catches anti-aliased stroke edges the channel-margin
    test alone would miss), so faint loop segments stay connected."""
    diff = cv2.absdiff(ann, cln).max(axis=2)
    drawn = diff > DIFF_THR

    b, g, r = ann[..., 0].astype(int), ann[..., 1].astype(int), ann[..., 2].astype(int)
    green_chan = (g > r + COLOR_MARGIN) & (g > b + COLOR_MARGIN)
    red_chan = (r > g + COLOR_MARGIN) & (r > b + COLOR_MARGIN)

    hsv = cv2.cvtColor(ann, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    saturated = (sat > 70) & (val > 60)
    red_hue = saturated & ((hue < 12) | (hue > 168))
    green_hue = saturated & (hue > 40) & (hue < 95)

    green = drawn & green_chan & green_hue
    red = drawn & red_chan & red_hue
    return green.astype(np.uint8) * 255, red.astype(np.uint8) * 255


def zone_from_green(green: np.ndarray) -> list[list[float]]:
    """Largest closed green loop → simplified polygon, normalized 0..1."""
    h, w = green.shape
    k = _odd(0.012 * max(h, w))
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SystemExit("no green zone stroke found")
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.0028 * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    return [[float(x / w), float(y / h)] for x, y in poly]


def _fill_loops(red: np.ndarray, close_k: int) -> np.ndarray:
    """Seal small gaps in the red strokes, then mark every enclosed interior.

    Flood-filling the background inward from a corner leaves exactly the pixels
    that a closed loop walls off; OR-ing those interiors back with the strokes
    yields one solid blob per drawn loop."""
    sealed = cv2.morphologyEx(red, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    h, w = sealed.shape
    flood = sealed.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)        # paint reachable background
    interiors = cv2.bitwise_not(flood)             # unreachable = walled off by a loop
    return cv2.bitwise_or(sealed, interiors)


def parts_from_red(red: np.ndarray) -> list[dict]:
    """Recover one part per drawn red loop, with bbox and polygon (normalized)."""
    h, w = red.shape
    image_area = float(h * w)
    close_k = _odd(0.0045 * max(h, w))             # seal pen gaps (~9px at 4k)
    open_k = _odd(0.0022 * max(h, w))              # split thin bridges between loops

    filled = _fill_loops(red, close_k)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * image_area
    max_area = MAX_AREA_FRAC * image_area

    parts: list[dict] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 4 or bh < 4:
            continue
        if area / float(bw * bh) < MIN_FILL_RATIO:  # noisy sliver, not a real loop
            continue
        eps = 0.012 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        parts.append({
            "bbox_norm": [x / w, y / h, (x + bw) / w, (y + bh) / h],
            "polygon_norm": [[float(px / w), float(py / h)] for px, py in poly],
            "area_px": area,
            "area_norm": area / image_area,
        })

    # Stable ids ordered top-to-bottom then left-to-right.
    parts.sort(key=lambda p: (round(p["bbox_norm"][1], 2), p["bbox_norm"][0]))
    for i, p in enumerate(parts):
        p["id"] = f"part_{i + 1:03d}"
    return parts


def render_overlay(clean: np.ndarray, zone: list[list[float]], parts: list[dict]) -> np.ndarray:
    """Verification overlay drawn purely from the extracted data: zone in green,
    each part polygon in a rotating palette so adjacent parts stay distinct."""
    vis = clean.copy()
    h, w = vis.shape[:2]
    palette = [
        (60, 60, 235), (40, 200, 255), (255, 170, 40),
        (220, 80, 220), (60, 220, 120), (235, 235, 60),
    ]
    pts = np.array([[int(x * w), int(y * h)] for x, y in zone], np.int32)
    cv2.polylines(vis, [pts], True, (0, 230, 0), max(2, int(0.0025 * max(h, w))))
    for i, p in enumerate(parts):
        color = palette[i % len(palette)]
        poly = np.array([[int(x * w), int(y * h)] for x, y in p["polygon_norm"]], np.int32)
        cv2.polylines(vis, [poly], True, color, max(2, int(0.0016 * max(h, w))))
        x1, y1 = int(p["bbox_norm"][0] * w), int(p["bbox_norm"][1] * h)
        cv2.putText(vis, str(i + 1), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


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

    vis = render_overlay(cln, zone, parts)
    vis_path = out.with_suffix(".check.jpg")
    cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out} and {vis_path}")


if __name__ == "__main__":
    main()
