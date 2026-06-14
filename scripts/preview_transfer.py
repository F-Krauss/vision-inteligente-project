#!/usr/bin/env python3
"""Eyeball annotation transfer: project reference part boxes onto candidate photos.

Runs the production ``transfer_annotations`` (ORB + ECC homography) and draws the
warped part boxes / polygons onto each candidate so you can visually confirm the
mapping lands on the real parts — especially the tiny ones. Per-candidate alignment
confidence is printed on the overlay; a low-confidence map is flagged in red so a
bad alignment is never mistaken for a good transfer.

Annotations are read from a registration JSON (``parts[].bbox_norm`` and/or
``polygon``/``polygon_norm``, plus optional ``zone_polygon_norm``) — the format
produced by scripts/extract_manual_annotation.py — or any JSON with a ``parts`` or
``annotations`` list of normalized boxes/polygons.

Usage:
  python scripts/preview_transfer.py \
      --reference anotated-example/IMG_2440.jpg \
      --registration anotated-example/IMG_2440_registration.json \
      --candidates path/to/other_same_angle_photo.jpg [more.jpg ...] \
      --out reports/transfer_preview

  # No candidates → projects onto the reference itself (sanity: boxes land exactly).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mold_inspection.piece_inspector import transfer_annotations  # noqa: E402

GREEN = (60, 200, 60)
RED = (40, 40, 230)
BLUE = (235, 160, 30)


def load_annotations(registration: dict) -> list[dict]:
    """Normalize a registration JSON into transfer_annotations' annotation dicts."""
    raw = registration.get("parts") or registration.get("annotations") or []
    annotations: list[dict] = []
    for i, part in enumerate(raw):
        ann: dict = {"id": part.get("id", f"part_{i}")}
        polygon = part.get("polygon") or part.get("polygon_norm")
        if isinstance(polygon, list) and len(polygon) >= 3:
            ann["polygon"] = [[float(p[0]), float(p[1])] for p in polygon]
        bbox = part.get("bbox") or part.get("bbox_norm")
        if isinstance(bbox, list) and len(bbox) == 4:
            ann["bbox"] = [float(v) for v in bbox]
        if "polygon" in ann or "bbox" in ann:
            annotations.append(ann)
    return annotations


def draw_annotations(image, annotations: list[dict], color) -> None:
    h, w = image.shape[:2]
    for ann in annotations:
        polygon = ann.get("polygon")
        if isinstance(polygon, list) and len(polygon) >= 3:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon], dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        elif isinstance(ann.get("bbox"), list):
            x1, y1, x2, y2 = ann["bbox"]
            cv2.rectangle(image, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), color, 2)


def draw_zone(image, zone_norm) -> None:
    if not (isinstance(zone_norm, list) and len(zone_norm) >= 3):
        return
    h, w = image.shape[:2]
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone_norm], dtype=np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=BLUE, thickness=3, lineType=cv2.LINE_AA)


def banner(image, text: str, ok: bool) -> None:
    color = GREEN if ok else RED
    cv2.rectangle(image, (0, 0), (image.shape[1], 46), (20, 20, 20), -1)
    cv2.putText(image, text, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--candidates", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=Path("reports/transfer_preview"))
    parser.add_argument("--min-confidence", type=float, default=0.08,
                        help="alignment inlier ratio below which a transfer is flagged unreliable")
    args = parser.parse_args()

    registration = json.loads(args.registration.read_text())
    annotations = load_annotations(registration)
    if not annotations:
        print("No parts/annotations found in registration JSON.", file=sys.stderr)
        return 2
    zone = registration.get("zone_polygon_norm")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(annotations)} parts from {args.registration.name}")

    reference_img = cv2.imread(str(args.reference))
    if reference_img is None:
        print(f"Could not read reference {args.reference} (HEIC? convert to JPG first).", file=sys.stderr)
        return 2

    # Reference overlay (the source-of-truth boxes).
    ref_overlay = reference_img.copy()
    draw_zone(ref_overlay, zone)
    draw_annotations(ref_overlay, annotations, GREEN)
    banner(ref_overlay, f"REFERENCE  {len(annotations)} parts", True)
    ref_out = args.out / f"{args.reference.stem}_reference_overlay.jpg"
    cv2.imwrite(str(ref_out), ref_overlay)
    print(f"  wrote {ref_out}")

    candidates = args.candidates or [args.reference]
    results = transfer_annotations(args.reference, candidates, annotations)
    for cand_path, result in zip(candidates, results):
        cand_img = cv2.imread(str(cand_path))
        if cand_img is None:
            print(f"  skip unreadable candidate {cand_path}", file=sys.stderr)
            continue
        overlay = cand_img.copy()
        ok = bool(result.get("ok")) and float(result.get("confidence", 0.0)) >= args.min_confidence
        warped = result.get("annotations") or []
        if ok:
            draw_annotations(overlay, warped, GREEN)
        msg = f"{cand_path.name}  conf={result.get('confidence', 0.0):.2f}  parts={len(warped)}"
        if not ok:
            msg += f"  UNRELIABLE: {result.get('message') or 'low alignment'}"
        banner(overlay, msg, ok)
        out = args.out / f"{cand_path.stem}_transfer_overlay.jpg"
        cv2.imwrite(str(out), overlay)
        print(f"  wrote {out}  (ok={ok}, conf={result.get('confidence', 0.0):.3f})")

    print(f"\nDone. Open the overlays in {args.out} to confirm boxes land on the parts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
