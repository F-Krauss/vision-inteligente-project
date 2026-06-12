#!/usr/bin/env python3
"""Run the per-part presence check for one section using its registration.

Flow (the product flow):
  1. Baseline: extra golden frames are aligned onto the primary golden and
     per-piece signals between them define benign jitter (fusion.golden_stats).
  2. Each eval photo is aligned onto the golden; every registered part gets
     signals -> fused score -> present / uncertain / missing.
  3. Verdicts are compared against the bootstrap ground truth
     (missing_piece_ids per eval image, mapped to registry ids by IoU).

Usage:
  python scripts/presence_check_registry.py \
      --slug try-photos-last-mold-with-without_pieces__vista_frontal__front_center_medium
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

from mold_inspection.presence.alignment import align_zone, warp_to_golden  # noqa: E402
from mold_inspection.presence.embedder import get_embedder  # noqa: E402
from mold_inspection.presence.fusion import decide, fuse, golden_stats  # noqa: E402
from mold_inspection.presence.imageio import clahe_gray, read_image  # noqa: E402
from mold_inspection.presence.signals import piece_signals  # noqa: E402

WORK_SIDE = 2000

COLORS = {"present": (0, 200, 0), "uncertain": (0, 200, 230), "missing": (0, 0, 235)}


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def grays(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return g, clahe_gray(g)


def signals_for_frame(golden, warped, pieces_px, embedder, g_gray, g_clahe):
    i_gray, i_clahe = grays(warped)
    out = {}
    for pid, bbox in pieces_px.items():
        out[pid] = piece_signals(
            golden, warped, {"bbox_px": bbox}, embedder,
            golden_gray=g_gray, golden_gray_clahe=g_clahe,
            insp_gray=i_gray, insp_gray_clahe=i_clahe,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", default="reports/presence_check")
    args = ap.parse_args()

    reg = json.loads((REPO_ROOT / "benchmarks/registrations" / f"{args.slug}.json").read_text())
    ann = json.loads((REPO_ROOT / "benchmarks/annotations" / f"{args.slug}.json").read_text())
    out_dir = REPO_ROOT / args.out / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    zone_poly = [tuple(pt) for pt in reg["zone_polygon_norm"]]
    golden = read_image(reg["golden_image"], max_side=WORK_SIDE)
    gh, gw = golden.shape[:2]
    g_gray, g_clahe = grays(golden)
    embedder = get_embedder()

    pieces_px = {p["id"]: (int(p["bbox_norm"][0] * gw), int(p["bbox_norm"][1] * gh),
                           int(p["bbox_norm"][2] * gw), int(p["bbox_norm"][3] * gh))
                 for p in reg["parts"]}

    # ── Baseline from extra golden frames ─────────────────────────────────────
    # Need >= 2 cross-golden samples to learn benign pose jitter; promote the
    # first present-only eval images into the baseline when registration is thin.
    per_piece_baseline: dict[str, list[dict]] = {pid: [] for pid in pieces_px}
    extra_goldens = [p for p in ann["golden_images"] if p != reg["golden_image"]]
    eval_entries = list(ann["eval_images"])
    while len(extra_goldens) < 3:
        promo = next((e for e in eval_entries if not e["missing_piece_ids"]), None)
        if promo is None:
            break
        eval_entries.remove(promo)
        extra_goldens.append(promo["path"])
        print(f"[baseline] promoted {Path(promo['path']).name} from eval to baseline")
    for path in extra_goldens:
        img = read_image(path, max_side=WORK_SIDE)
        al = align_zone(golden, img, zone_poly, max_side=1600)
        if not al.ok:
            print(f"[baseline] {Path(path).name}: alignment failed, skipped")
            continue
        warped = warp_to_golden(img, al, golden.shape)
        sig = signals_for_frame(golden, warped, pieces_px, embedder, g_gray, g_clahe)
        for pid, s in sig.items():
            per_piece_baseline[pid].append(s)
        print(f"[baseline] {Path(path).name}: ok (inliers {al.inlier_ratio:.2f})")

    stats = {}
    for pid, samples in per_piece_baseline.items():
        if not samples:  # no cross-golden sample: benign-jitter unknown, use self with wide floors
            print(f"[baseline] WARNING {pid}: no cross-golden sample")
            samples = [signals_for_frame(golden, golden, {pid: pieces_px[pid]}, embedder, g_gray, g_clahe)[pid]]
        stats[pid] = golden_stats(samples)

    # ── Ground truth mapping: bootstrap piece ids -> registry ids by IoU ─────
    bootstrap_pieces = [p for p in ann["pieces"] if not p.get("control")]
    registry_ids = {p["id"] for p in reg["parts"]}
    gt_map = {}
    for bp in bootstrap_pieces:
        if bp["id"] in registry_ids:
            gt_map[bp["id"]] = bp["id"]
            continue
        bx = (bp["bbox_norm"][0] + bp["bbox_norm"][2]) / 2
        by = (bp["bbox_norm"][1] + bp["bbox_norm"][3]) / 2
        best, best_key = None, 0.0
        for rp in reg["parts"]:
            x1, y1, x2, y2 = rp["bbox_norm"]
            inside = x1 <= bx <= x2 and y1 <= by <= y2
            v = iou(bp["bbox_norm"], rp["bbox_norm"])
            key = v + (0.5 if inside else 0.0)
            if key > best_key:
                best, best_key = rp["id"], key
        if best and best_key >= 0.05:
            gt_map[bp["id"]] = best
    print(f"ground-truth map: {gt_map}")

    # ── Evaluate every image ──────────────────────────────────────────────────
    rows = []
    for entry in eval_entries:
        path = entry["path"]
        name = Path(path).name
        expected = {gt_map.get(i) for i in entry["missing_piece_ids"] if gt_map.get(i)}
        img = read_image(path, max_side=WORK_SIDE)
        al = align_zone(golden, img, zone_poly, max_side=1600)
        if not al.ok:
            rows.append({"image": name, "expected": sorted(expected), "error": "alignment_failed"})
            print(f"[eval] {name}: ALIGNMENT FAILED")
            continue
        warped = warp_to_golden(img, al, golden.shape)
        sig = signals_for_frame(golden, warped, pieces_px, embedder, g_gray, g_clahe)

        verdicts = {}
        for pid, s in sig.items():
            score = fuse(s, stats[pid])
            verdicts[pid] = {"score": round(score, 3), "decision": decide(score)}

        detected = sorted(pid for pid, v in verdicts.items() if v["decision"] == "missing")
        uncertain = sorted(pid for pid, v in verdicts.items() if v["decision"] == "uncertain")
        tp = sorted(set(detected) & expected)
        fp = sorted(set(detected) - expected)
        fn = sorted(expected - set(detected))
        rows.append({"image": name, "expected": sorted(expected), "detected_missing": detected,
                     "uncertain": uncertain, "tp": tp, "fp": fp, "fn": fn,
                     "verdicts": verdicts})
        print(f"[eval] {name}: expected={sorted(expected) or '-'} detected={detected or '-'} "
              f"uncertain={uncertain or '-'}")

        # Render
        vis = golden.copy()
        zp = np.array([[int(x * gw), int(y * gh)] for x, y in reg["zone_polygon_norm"]], np.int32)
        cv2.polylines(vis, [zp], True, (255, 200, 0), 3)
        vis = cv2.addWeighted(warped, 0.65, vis, 0.35, 0)
        for pid, v in verdicts.items():
            x1, y1, x2, y2 = pieces_px[pid]
            c = COLORS[v["decision"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 3 if v["decision"] != "present" else 2)
            if v["decision"] != "present":
                cv2.putText(vis, f'{pid} {v["score"]:.2f}', (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
        cv2.imwrite(str(out_dir / f"check_{name}"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    ok_rows = [r for r in rows if "error" not in r]
    n_tp = sum(len(r["tp"]) for r in ok_rows)
    n_fp = sum(len(r["fp"]) for r in ok_rows)
    n_fn = sum(len(r["fn"]) for r in ok_rows)
    n_expected = sum(len(r["expected"]) for r in ok_rows)
    present_imgs = [r for r in ok_rows if not r["expected"]]
    clean_present = sum(1 for r in present_imgs if not r["detected_missing"])
    print("\n=== Summary ===")
    print(f"eval images: {len(rows)} ({len(rows) - len(ok_rows)} alignment failures)")
    print(f"expected missing instances: {n_expected}  detected TP={n_tp}  FN={n_fn}  FP={n_fp}")
    print(f"present-only images with zero false missing: {clean_present}/{len(present_imgs)}")
    print(f"renders + results.json -> {out_dir}")


if __name__ == "__main__":
    main()
