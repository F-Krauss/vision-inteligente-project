from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


# Inference resolution for the per-part YOLO detector. Parts can be ~15px on a
# ~4032px frame; ultralytics' default 640 downscales those to ~2px (invisible), so
# we predict at a much larger size and (on by default) tile the full-res frame.
PIECE_DETECTOR_IMGSZ = 1280
PIECE_DETECTOR_CONF = 0.35
_TILE_OVERLAP = 0.2  # fraction of tile size overlapped between adjacent tiles
_TILE_MERGE_IOU = 0.5  # boxes above this IoU across tiles are the same detection


def inspect_expected_pieces(
    family: str,
    zone_id: str,
    image_path: str | Path,
    datasets: list[dict[str, Any]],
    registry_dir: str | Path = "data/model_registry",
    confidence: float = PIECE_DETECTOR_CONF,
    expected_pieces: list[dict[str, Any]] | None = None,
    imgsz: int = PIECE_DETECTOR_IMGSZ,
    tile: bool = True,
    detector_path: str | Path | None = None,
) -> dict[str, Any]:
    expected = expected_pieces or _expected_pieces(family, zone_id, datasets)
    if not expected:
        return {"status": "not_configured", "findings": [], "message": "No hay piezas esperadas configuradas."}

    detector = Path(detector_path) if detector_path else Path(registry_dir) / family / zone_id / "piece_detector" / "best.pt"
    if not detector.exists():
        return {
            "status": "review",
            "reason": "missing_piece_detector",
            "message": "Hay piezas esperadas, pero falta entrenar el detector especializado de piezas.",
            "findings": [
                {
                    "piece_id": piece["id"],
                    "class_name": piece["class_name"],
                    "status": "uncertain",
                    "confidence": 0.0,
                    "region": _piece_region(piece),
                }
                for piece in expected
                if piece.get("required", True)
            ],
        }

    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "status": "review",
            "reason": "missing_dependency",
            "message": "Falta ultralytics para ejecutar el detector especializado de piezas.",
            "findings": [],
        }

    detections, orig_w, orig_h = predict_piece_detections(detector, image_path, confidence=confidence, imgsz=imgsz, tile=tile)

    required = [piece for piece in expected if piece.get("required", True)]
    assigned_piece, consumed_dets = _assign_detections_to_pieces(detections, required, orig_w, orig_h)

    findings = []
    missing = 0
    for piece in required:
        match = assigned_piece.get(piece["id"])
        if match:
            findings.append(
                {
                    "piece_id": piece["id"],
                    "class_name": piece["class_name"],
                    "status": "present",
                    "confidence": match["confidence"],
                    "bbox": match["bbox"],
                }
            )
        else:
            missing += 1
            findings.append(
                {
                    "piece_id": piece["id"],
                    "class_name": piece["class_name"],
                    "status": "missing",
                    "confidence": 0.0,
                    "region": _piece_region(piece),
                }
            )

    # Detections that matched no expected slot are unexpected/extra parts. Reported
    # for identification but kept out of the pass/fail gate (the detector can fire on
    # background), so this never causes a false reject on its own.
    unexpected = [det for det in detections if det["det_index"] not in consumed_dets]
    for det in unexpected:
        findings.append(
            {
                "piece_id": f"unexpected_{det['det_index']}",
                "class_name": det["class_name"],
                "status": "unexpected",
                "confidence": det["confidence"],
                "bbox": det["bbox"],
            }
        )

    return {
        "status": "correct" if missing == 0 else "review",
        "message": "Todas las piezas esperadas fueron detectadas." if missing == 0 else "Faltan piezas esperadas.",
        "findings": findings,
        "missing_count": missing,
        "unexpected_count": len(unexpected),
    }


def predict_piece_detections(
    model_path: str | Path,
    image_path: str | Path,
    confidence: float = PIECE_DETECTOR_CONF,
    imgsz: int = PIECE_DETECTOR_IMGSZ,
    tile: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install vision dependencies: python -m pip install -e ".[vision]"') from exc

    model = YOLO(str(model_path))
    if tile:
        return _tiled_predict(model, str(image_path), imgsz, confidence)
    result = model.predict(str(image_path), imgsz=imgsz, conf=confidence, verbose=False)[0]
    names = result.names or {}
    orig_h, orig_w = (result.orig_shape if getattr(result, "orig_shape", None) else (0, 0))
    detections = []
    for index, box in enumerate(result.boxes or []):
        class_id = int(box.cls[0])
        detections.append(
            {
                "det_index": index,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    return detections, int(orig_w), int(orig_h)


def _tiled_predict(
    model: Any,
    image_path: str,
    imgsz: int,
    conf: float,
    overlap: float = _TILE_OVERLAP,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run the detector over overlapping full-resolution tiles and merge.

    Predicting a 4032px frame at imgsz downscales tiny parts to nothing. Instead we
    slice the frame into ~imgsz-sized tiles, predict each at native resolution (so a
    15px part stays 15px), offset the boxes back to full-frame pixels, and merge
    duplicates from overlapping tiles with class-aware IoU-NMS. Returns the same
    detection dicts as the single-shot path plus the full-frame (orig_w, orig_h).
    """
    if cv2 is None:
        result = model.predict(image_path, imgsz=imgsz, conf=conf, verbose=False)[0]
        names = result.names or {}
        orig_h, orig_w = (result.orig_shape if getattr(result, "orig_shape", None) else (0, 0))
        dets = [
            {
                "det_index": index,
                "class_name": str(names.get(int(box.cls[0]), int(box.cls[0]))),
                "confidence": float(box.conf[0]),
                "bbox": [float(v) for v in box.xyxy[0].tolist()],
            }
            for index, box in enumerate(result.boxes or [])
        ]
        return dets, int(orig_w), int(orig_h)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return [], 0, 0
    height, width = image.shape[:2]
    tile_px = max(320, int(imgsz))
    step = max(1, int(tile_px * (1.0 - overlap)))

    xs = _tile_origins(width, tile_px, step)
    ys = _tile_origins(height, tile_px, step)

    raw: list[dict[str, Any]] = []
    for y0 in ys:
        for x0 in xs:
            crop = image[y0:y0 + tile_px, x0:x0 + tile_px]
            if crop.size == 0:
                continue
            result = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)[0]
            names = result.names or {}
            for box in result.boxes or []:
                bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0].tolist())
                raw.append(
                    {
                        "class_name": str(names.get(int(box.cls[0]), int(box.cls[0]))),
                        "confidence": float(box.conf[0]),
                        "bbox": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                    }
                )

    merged = _merge_tile_detections(raw, _TILE_MERGE_IOU)
    for index, det in enumerate(merged):
        det["det_index"] = index
    return merged, width, height


def _tile_origins(extent: int, tile_px: int, step: int) -> list[int]:
    """Tile start offsets covering [0, extent), the last tile flush to the edge."""
    if extent <= tile_px:
        return [0]
    origins = list(range(0, extent - tile_px + 1, step))
    last = extent - tile_px
    if origins[-1] != last:
        origins.append(last)
    return origins


def _merge_tile_detections(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """Class-aware greedy NMS: keep the highest-confidence box, drop same-class boxes
    overlapping it above ``iou_threshold`` (the same physical part seen in two tiles)."""
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        if any(
            other["class_name"] == det["class_name"]
            and _xyxy_iou(other["bbox"], det["bbox"]) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(det)
    return kept


def _xyxy_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def inspect_expected_pieces_against_reference(
    family: str,
    zone_id: str,
    image_path: str | Path,
    reference_image_path: str | Path,
    expected_pieces: list[dict[str, Any]],
    evidence_dir: str | Path | None = None,
    difference_threshold: int = 34,
    min_changed_fraction: float = 0.025,
    min_mean_delta: float = 7.0,
) -> dict[str, Any]:
    expected = [piece for piece in expected_pieces if piece.get("required", True)]
    if not expected:
        return {"status": "not_configured", "findings": [], "message": "No hay piezas esperadas configuradas."}
    if cv2 is None or np is None:
        return {
            "status": "review",
            "reason": "missing_dependency",
            "message": "Falta OpenCV para comparar piezas contra referencia golden.",
            "findings": [],
        }

    candidate = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(reference_image_path), cv2.IMREAD_COLOR)
    if candidate is None or reference is None:
        return {
            "status": "review",
            "reason": "unreadable_reference_pair",
            "message": "No se pudo leer la captura o referencia golden para comparar piezas.",
            "findings": [],
        }
    height, width = candidate.shape[:2]
    if reference.shape[:2] != (height, width):
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)

    candidate, alignment = _align_candidate_to_reference(reference, candidate)
    valid_mask = alignment.pop("_valid_mask", None)
    candidate_gray = cv2.GaussianBlur(cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    reference_gray = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(candidate_gray, reference_gray)
    if valid_mask is not None:
        diff[valid_mask == 0] = 0

    findings: list[dict[str, Any]] = []
    localized_findings: list[dict[str, Any]] = []
    uncertain_count = 0
    overlay = candidate.copy()
    for piece in expected:
        region = _piece_region(piece)
        x1, y1, x2, y2 = _roi_bounds(piece, width, height)
        crop = diff[y1:y2, x1:x2]
        if crop.size == 0:
            findings.append(
                {
                    "piece_id": piece["id"],
                    "class_name": piece["class_name"],
                    "status": "uncertain",
                    "confidence": 0.0,
                    "region": region,
                    "reason": "empty_roi",
                }
            )
            continue
        changed_fraction = float(np.count_nonzero(crop > difference_threshold) / crop.size)
        mean_delta = float(crop.mean())
        p95_delta = float(np.percentile(crop, 95))
        change_score = _change_score(mean_delta, changed_fraction)
        localized = _localized_changes(
            crop,
            x1,
            y1,
            width,
            height,
            difference_threshold=difference_threshold,
            min_mean_delta=min_mean_delta,
        )
        if localized:
            for index, change in enumerate(localized, start=1):
                localized_findings.append(
                    {
                        "piece_id": f"{piece['id']}_change_{index}",
                        "expected_piece_id": piece["id"],
                        "class_name": piece["class_name"],
                        "status": "missing",
                        "confidence": round(max(change_score, change["confidence"]), 4),
                        "region": change["region"],
                        "bbox_normalized": change["bbox_normalized"],
                        "method": "reference_localized_diff",
                        "change_score": round(max(change_score, change["confidence"]), 4),
                        "changed_fraction": round(changed_fraction, 4),
                        "localized_changed_fraction": round(change["changed_fraction"], 4),
                        "mean_delta": round(mean_delta, 4),
                        "localized_mean_delta": round(change["mean_delta"], 4),
                        "p95_delta": round(p95_delta, 4),
                        "_contour": change["_contour"],
                        "_roi_area": _normalized_roi_area(piece),
                    }
                )
        else:
            status = "present"
            reason = None
            if changed_fraction >= min_changed_fraction and mean_delta >= min_mean_delta:
                status = "uncertain"
                reason = "global_change_without_localized_piece"
                uncertain_count += 1
            finding = {
                "piece_id": piece["id"],
                "class_name": piece["class_name"],
                "status": status,
                "confidence": round(1.0 - min(change_score, 0.95), 4),
                "region": region,
                "method": "reference_localized_diff",
                "change_score": round(change_score, 4),
                "changed_fraction": round(changed_fraction, 4),
                "mean_delta": round(mean_delta, 4),
                "p95_delta": round(p95_delta, 4),
            }
            if reason:
                finding["reason"] = reason
            findings.append(finding)

    localized_findings = _dedupe_localized_findings(localized_findings)
    for finding in localized_findings:
        contour = finding.pop("_contour", None)
        finding.pop("_roi_area", None)
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, (20, 20, 230), 4)
            x, y, w, h = cv2.boundingRect(contour)
        else:
            bbox = finding["bbox_normalized"]
            x = int(round(float(bbox["x"]) * width))
            y = int(round(float(bbox["y"]) * height))
            w = int(round(float(bbox["width"]) * width))
            h = int(round(float(bbox["height"]) * height))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 230), 4)
        center = (int(x + w / 2), int(y + h / 2))
        radius = max(16, int(max(w, h) * 0.72))
        cv2.circle(overlay, center, radius, (20, 20, 230), 3)
    findings.extend(localized_findings)

    missing = len(localized_findings)
    status = "correct" if missing == 0 and uncertain_count == 0 else "review"
    if missing:
        message = "Cambios localizados en piezas esperadas contra golden sample."
    elif uncertain_count:
        message = "Hay cambios globales de captura; no se localizo pieza faltante."
    else:
        message = "Piezas esperadas coinciden con golden sample."

    overlay_image = None
    if evidence_dir:
        output_dir = Path(evidence_dir) / family / zone_id / Path(image_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = output_dir / "reference_roi_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        overlay_image = str(overlay_path)

    return {
        "status": status,
        "reason": "reference_roi_diff",
        "message": message,
        "findings": findings,
        "missing_count": missing,
        "uncertain_count": uncertain_count,
        "alignment": alignment,
        "overlay_image": overlay_image,
    }


def inspect_expected_pieces_against_references(
    family: str,
    zone_id: str,
    candidate_image_path: str | Path,
    annotated_references: list[dict[str, Any]],
    evidence_dir: str | Path | None = None,
    min_support: int | None = None,
    **diff_kwargs: Any,
) -> dict[str, Any] | None:
    """Per-part consensus across multiple annotated golden images.

    ``annotated_references`` is ``[{"image_path": <local path>, "boxes":
    [{element_id, class_name, bbox}]}]``. A canonical part set is the union of all
    references' boxes (by ``element_id``); each reference then casts one vote per
    part via the existing ECC-aligned per-ROI diff (``inspect_expected_pieces_
    against_reference``). Votes are combined with a false-rejection-safe three-state
    rule so a single bad/occluded golden can never silently approve a part:

      • flagged ``missing`` by a strict majority of references → ``missing``;
      • flagged by some but not a majority (or any inconclusive) → ``uncertain`` → review;
      • flagged by none → ``present``.

    Returns the pipeline-compatible piece_inspection dict, or ``None`` when no
    reference could be compared (so the caller falls back to a coarser path).
    """
    refs = [r for r in annotated_references if r.get("image_path") and r.get("boxes")]
    if not refs:
        return None
    canonical = _canonical_pieces_from_refs(refs)
    if not canonical:
        return None

    votes: list[dict[str, dict[str, Any]]] = []
    overlays: list[str] = []
    for ref in refs:
        try:
            res = inspect_expected_pieces_against_reference(
                family=family,
                zone_id=zone_id,
                image_path=candidate_image_path,
                reference_image_path=ref["image_path"],
                expected_pieces=canonical,
                evidence_dir=evidence_dir,
                **diff_kwargs,
            )
        except Exception:  # pragma: no cover - defensive
            continue
        if res.get("reason") in {"missing_dependency", "unreadable_reference_pair", "not_configured"}:
            continue  # this reference could not be compared
        votes.append(_piece_status_map(res, canonical))
        if res.get("overlay_image"):
            overlays.append(str(res["overlay_image"]))

    if not votes:
        return None

    n_refs = len(votes)
    required = min_support if min_support is not None else (n_refs // 2 + 1)
    required = max(1, min(required, n_refs))

    findings: list[dict[str, Any]] = []
    missing = 0
    uncertain = 0
    for piece in canonical:
        pid = piece["id"]
        entries = [vote.get(pid, {"status": "present"}) for vote in votes]
        miss_votes = [entry for entry in entries if entry["status"] == "missing"]
        unc_votes = sum(1 for entry in entries if entry["status"] == "uncertain")
        support = len(miss_votes)
        bbox_normalized = next((entry.get("bbox_normalized") for entry in miss_votes if entry.get("bbox_normalized")), None)
        # Prefer the tight localized change box (from a flagging reference) over the
        # broad ROI polygon, so the reported region pinpoints the missing part.
        region = _bbox_to_region(bbox_normalized) if bbox_normalized else _piece_region(piece)
        if support >= required:
            status = "missing"
            missing += 1
        elif support > 0 or unc_votes > 0:
            status = "uncertain"
            uncertain += 1
        else:
            status = "present"
        finding = {
            "piece_id": pid,
            "class_name": piece["class_name"],
            "status": status,
            "confidence": round(support / n_refs, 4),
            "region": region,
            "support": support,
            "support_ratio": round(support / n_refs, 3),
            "uncertain_votes": unc_votes,
            "method": "reference_consensus",
        }
        if bbox_normalized:
            finding["bbox_normalized"] = bbox_normalized
        findings.append(finding)

    status = "correct" if (missing == 0 and uncertain == 0) else "review"
    if missing:
        message = f"{missing} pieza(s) faltante(s) confirmada(s) por ≥{required} de {n_refs} referencias."
    elif uncertain:
        message = "Cambios no concluyentes contra referencias; requiere revisión humana."
    else:
        message = "Todas las piezas presentes — confirmado contra todas las referencias."

    return {
        "status": status,
        "reason": "reference_roi_consensus",
        "message": message,
        "findings": findings,
        "missing_count": missing,
        "uncertain_count": uncertain,
        "reference_count": n_refs,
        "required_support": required,
        "overlay_image": overlays[0] if overlays else None,
        "method": "reference_consensus",
    }


def _bbox_to_region(bbox: dict[str, float]) -> list[dict[str, float]]:
    bx, by = float(bbox["x"]), float(bbox["y"])
    bw, bh = float(bbox["width"]), float(bbox["height"])
    return [
        {"x": bx, "y": by},
        {"x": bx + bw, "y": by},
        {"x": bx + bw, "y": by + bh},
        {"x": bx, "y": by + bh},
    ]


def _canonical_pieces_from_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of all references' boxes keyed by element_id → expected-piece dicts."""
    by_id: dict[str, dict[str, Any]] = {}
    for ref in refs:
        for box in ref.get("boxes") or []:
            element_id = str(box.get("element_id") or box.get("class_name") or "piece")
            if element_id in by_id:
                continue
            bbox = box.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            by_id[element_id] = {
                "id": element_id,
                "class_name": str(box.get("class_name") or "piece"),
                "roi": [float(value) for value in bbox],
                "required": True,
            }
    return list(by_id.values())


def _piece_status_map(result: dict[str, Any], canonical: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One reference's per-part verdict: piece_id → {status, bbox_normalized?}.

    A part the single-reference diff did not flag is ``present`` in that reference;
    localized changes (carried under ``expected_piece_id``) mark it ``missing``."""
    statuses: dict[str, dict[str, Any]] = {piece["id"]: {"status": "present"} for piece in canonical}
    for finding in result.get("findings", []):
        pid = finding.get("expected_piece_id") or finding.get("piece_id")
        if pid not in statuses:
            continue
        finding_status = finding.get("status")
        if finding_status == "missing":
            statuses[pid] = {
                "status": "missing",
                "bbox_normalized": finding.get("bbox_normalized"),
                "confidence": finding.get("confidence", 0.0),
            }
        elif finding_status == "uncertain" and statuses[pid]["status"] == "present":
            statuses[pid] = {"status": "uncertain"}
    return statuses


def transfer_annotations(
    reference_path: str | Path,
    candidate_paths: list[str | Path],
    annotations: list[dict[str, Any]],
    min_inlier_ratio: float = 0.08,
) -> list[dict[str, Any]]:
    """Project reference annotations onto each candidate photo via ORB homography.

    Each annotation carries normalized ``polygon`` (and/or ``bbox``) in the
    reference frame; we align the candidate to the reference, invert the
    homography to map reference→candidate, and warp every vertex. Returns, per
    candidate, the warped annotations plus an alignment confidence so the UI can
    flag low-quality maps for manual review."""
    results: list[dict[str, Any]] = []
    if cv2 is None or np is None:
        for path in candidate_paths:
            results.append({"path": str(path), "ok": False, "confidence": 0.0,
                            "message": "Falta OpenCV para mapear anotaciones.", "annotations": []})
        return results

    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        for path in candidate_paths:
            results.append({"path": str(path), "ok": False, "confidence": 0.0,
                            "message": "No se pudo leer la imagen de referencia.", "annotations": []})
        return results
    rh, rw = reference.shape[:2]

    for path in candidate_paths:
        candidate = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if candidate is None:
            results.append({"path": str(path), "ok": False, "confidence": 0.0,
                            "message": "No se pudo leer la imagen de comparación.", "annotations": []})
            continue
        if candidate.shape[:2] != (rh, rw):
            candidate = cv2.resize(candidate, (rw, rh), interpolation=cv2.INTER_AREA)

        _, alignment = _align_candidate_to_reference(reference, candidate)
        homography = alignment.get("_homography")
        confidence = float(alignment.get("inlier_ratio") or 0.0)
        if not alignment.get("ok") or homography is None or confidence < min_inlier_ratio:
            results.append({"path": str(path), "ok": False, "confidence": round(confidence, 4),
                            "message": alignment.get("reason") or "Alineación insuficiente.", "annotations": []})
            continue

        try:
            ref_to_candidate = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            results.append({"path": str(path), "ok": False, "confidence": round(confidence, 4),
                            "message": "Homografía no invertible.", "annotations": []})
            continue

        warped_annotations = [_warp_annotation(ann, ref_to_candidate, rw, rh) for ann in annotations]
        results.append({"path": str(path), "ok": True, "confidence": round(confidence, 4),
                        "message": "", "annotations": warped_annotations})
    return results


def _warp_annotation(annotation: dict[str, Any], matrix: Any, width: int, height: int) -> dict[str, Any]:
    polygon = annotation.get("polygon")
    if not (isinstance(polygon, list) and len(polygon) >= 2):
        bbox = annotation.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    pts = np.array([[[float(p[0]) * width, float(p[1]) * height]] for p in polygon], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    new_polygon = [[_clip01(px / width), _clip01(py / height)] for px, py in warped]
    xs = [p[0] for p in new_polygon]
    ys = [p[1] for p in new_polygon]
    new_bbox = [min(xs), min(ys), max(xs), max(ys)]
    return {**annotation, "polygon": new_polygon, "bbox": new_bbox}


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _expected_pieces(family: str, zone_id: str, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in reversed(datasets):
        source = record.get("data") if isinstance(record.get("data"), dict) else record
        if source.get("family") == family and source.get("zone_id") == zone_id:
            pieces = source.get("expected_pieces") or []
            if isinstance(pieces, list):
                return [piece for piece in pieces if isinstance(piece, dict)]
    return []


def _piece_region(piece: dict[str, Any]) -> list[dict[str, float]] | None:
    region = piece.get("region") or piece.get("polygon")
    if isinstance(region, list) and region:
        return region
    roi = piece.get("roi")
    if not (isinstance(roi, list) and len(roi) == 4):
        return None
    x1, y1, x2, y2 = [float(value) for value in roi]
    return [
        {"x": x1, "y": y1},
        {"x": x2, "y": y1},
        {"x": x2, "y": y2},
        {"x": x1, "y": y2},
    ]


def _roi_bounds(piece: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    roi = piece.get("roi")
    if not (isinstance(roi, list) and len(roi) == 4):
        return 0, 0, width, height
    x1, y1, x2, y2 = [float(value) for value in roi]
    left = max(0, min(width - 1, int(round(x1 * width))))
    top = max(0, min(height - 1, int(round(y1 * height))))
    right = max(left + 1, min(width, int(round(x2 * width))))
    bottom = max(top + 1, min(height, int(round(y2 * height))))
    return left, top, right, bottom


def _assign_detections_to_pieces(
    detections: list[dict[str, Any]],
    required: list[dict[str, Any]],
    orig_w: int,
    orig_h: int,
) -> tuple[dict[str, dict[str, Any]], set[int]]:
    """Greedily assign detections to expected pieces by class AND location, consuming
    each detection at most once. Without consumption + location, N expected parts of
    the same class all resolve to a single same-class detection — so a specific
    missing part (e.g. one of many identical screws) could never be flagged while any
    survived. Highest (confidence × overlap) pairs win first. Returns the per-piece
    assigned detection and the set of consumed detection indices."""
    candidate_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for piece in required:
        roi_px = _roi_bounds(piece, orig_w, orig_h) if orig_w and orig_h else None
        for det in detections:
            if det["class_name"] != piece["class_name"]:
                continue
            # When ROI/image size is unknown, fall back to class-only (overlap=1).
            overlap = 1.0 if roi_px is None else _roi_match_score(det["bbox"], roi_px)
            if overlap <= 0.0:
                continue
            candidate_pairs.append((det["confidence"] * (0.5 + 0.5 * overlap), piece, det))

    candidate_pairs.sort(key=lambda item: item[0], reverse=True)
    assigned_piece: dict[str, dict[str, Any]] = {}
    consumed_dets: set[int] = set()
    for _score, piece, det in candidate_pairs:
        if piece["id"] in assigned_piece or det["det_index"] in consumed_dets:
            continue
        assigned_piece[piece["id"]] = det
        consumed_dets.add(det["det_index"])
    return assigned_piece, consumed_dets


def _roi_match_score(det_bbox: list[float], roi_px: tuple[int, int, int, int]) -> float:
    """Spatial agreement between a detection box and an expected-piece ROI, both in
    pixels. Returns IoU, but treats a detection whose centre falls inside the ROI as
    a match too (parts are often small relative to a generous hand-drawn ROI)."""
    dx1, dy1, dx2, dy2 = det_bbox
    rx1, ry1, rx2, ry2 = roi_px
    inter_x1 = max(dx1, rx1)
    inter_y1 = max(dy1, ry1)
    inter_x2 = min(dx2, rx2)
    inter_y2 = min(dy2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    det_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
    roi_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = det_area + roi_area - intersection
    iou = intersection / union if union > 0 else 0.0
    if iou > 0:
        return iou
    cx = (dx1 + dx2) / 2.0
    cy = (dy1 + dy2) / 2.0
    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
        return 0.05  # weak-but-positive: centre inside ROI though boxes barely overlap
    return 0.0


def _change_score(mean_delta: float, changed_fraction: float) -> float:
    mean_component = min(mean_delta / 42.0, 1.0)
    area_component = min(changed_fraction / 0.22, 1.0)
    return (0.55 * mean_component) + (0.45 * area_component)


def _align_candidate_to_reference(reference: Any, candidate: Any) -> tuple[Any, dict[str, Any]]:
    height, width = reference.shape[:2]
    alignment: dict[str, Any] = {
        "method": "orb_homography",
        "ok": False,
        "matched_keypoints": 0,
        "inlier_ratio": 0.0,
    }
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=8000)
    ref_keypoints, ref_desc = orb.detectAndCompute(ref_gray, None)
    cand_keypoints, cand_desc = orb.detectAndCompute(cand_gray, None)
    if ref_desc is None or cand_desc is None:
        alignment["reason"] = "insufficient_features"
        return candidate, alignment

    # Lowe ratio test on KNN matches keeps only well-separated correspondences,
    # giving RANSAC far cleaner inliers than crossCheck top-N. Fall back to
    # crossCheck if the ratio test is too aggressive on this pair.
    good_matches = _ratio_matched(ref_desc, cand_desc)
    if len(good_matches) < 10:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(ref_desc, cand_desc), key=lambda item: item.distance)
        good_matches = matches[: min(320, len(matches))]
    alignment["matched_keypoints"] = len(good_matches)
    if len(good_matches) < 10:
        alignment["reason"] = "insufficient_matches"
        return candidate, alignment

    ref_points = np.float32([ref_keypoints[item.queryIdx].pt for item in good_matches]).reshape(-1, 1, 2)
    cand_points = np.float32([cand_keypoints[item.trainIdx].pt for item in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(cand_points, ref_points, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        alignment["reason"] = "homography_failed"
        return candidate, alignment

    inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))
    alignment["inlier_ratio"] = round(inlier_ratio, 4)
    if inlier_ratio < 0.08:
        alignment["reason"] = "low_inlier_ratio"
        return candidate, alignment

    # Sub-pixel polish: ECC homography refinement on top of the ORB estimate. This
    # is what makes ~15px parts land — feature-matched homographies on a 3D mold
    # routinely leave 5-20px residual; ECC drives the photometric alignment to
    # sub-pixel where texture supports it, falling back to ORB if it diverges.
    homography, ecc_info = _refine_homography_ecc(ref_gray, cand_gray, homography)
    alignment["ecc"] = ecc_info
    if ecc_info.get("applied"):
        alignment["method"] = "orb+ecc"

    aligned = cv2.warpPerspective(candidate, homography, (width, height))
    source_mask = np.full(candidate.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(source_mask, homography, (width, height))
    valid_mask = cv2.threshold(valid_mask, 250, 255, cv2.THRESH_BINARY)[1]
    valid_mask = _erode_valid_mask(valid_mask)
    alignment["ok"] = True
    alignment["valid_mask_ratio"] = round(float(np.count_nonzero(valid_mask)) / float(width * height), 4)
    alignment["_valid_mask"] = valid_mask
    # candidate→reference homography in reference-sized pixel space; the transfer
    # path inverts it to project reference annotations onto the candidate.
    alignment["_homography"] = homography
    return aligned, alignment


def _ratio_matched(ref_desc: Any, cand_desc: Any, ratio: float = 0.75) -> list[Any]:
    """ORB descriptor matches surviving Lowe's ratio test (best vs 2nd-best)."""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good: list[Any] = []
    for pair in matcher.knnMatch(ref_desc, cand_desc, k=2):
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    return good


def _refine_homography_ecc(
    ref_gray: Any,
    cand_gray: Any,
    homography: Any,
    max_side: int = 1280,
) -> tuple[Any, dict[str, Any]]:
    """Polish a candidate→reference homography to sub-pixel accuracy with ECC.

    ECC maximizes the Enhanced Correlation Coefficient (invariant to brightness/
    contrast), so it refines alignment under lighting changes where intensity diff
    would not. Runs on a downscaled grayscale pair for speed, then rescales the
    refined warp back to full reference resolution. Returns the original homography
    unchanged if ECC fails to converge or visibly diverges."""
    info: dict[str, Any] = {"applied": False}
    try:
        h, w = ref_gray.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        if scale < 1.0:
            size = (int(round(w * scale)), int(round(h * scale)))
            ref_s = cv2.resize(ref_gray, size, interpolation=cv2.INTER_AREA)
            cand_s = cv2.resize(cand_gray, size, interpolation=cv2.INTER_AREA)
        else:
            ref_s, cand_s = ref_gray, cand_gray
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        scale_inv = np.array([[1.0 / scale, 0, 0], [0, 1.0 / scale, 0], [0, 0, 1]], dtype=np.float64)
        # ECC's warp maps template(reference)→input(candidate), i.e. inv(homography).
        warp_full = np.linalg.inv(homography.astype(np.float64))
        warp_small = (scale_mat @ warp_full @ scale_inv).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        ref_b = cv2.GaussianBlur(ref_s, (5, 5), 0)
        cand_b = cv2.GaussianBlur(cand_s, (5, 5), 0)
        cc, warp_small = cv2.findTransformECC(ref_b, cand_b, warp_small, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
        warp_full_refined = scale_inv @ warp_small.astype(np.float64) @ scale_mat
        refined = np.linalg.inv(warp_full_refined)
        refined = refined / refined[2, 2]
        if not np.all(np.isfinite(refined)) or not _homography_close(homography, refined, w, h):
            info["reason"] = "ecc_diverged"
            return homography, info
        info.update({"applied": True, "cc": round(float(cc), 4)})
        return refined, info
    except cv2.error:
        info["reason"] = "ecc_failed"
        return homography, info


def _homography_close(h1: Any, h2: Any, w: int, h: int, max_frac: float = 0.06) -> bool:
    """True when two homographies map the image corners to within max_frac of the
    image's larger side — a guard so a diverged ECC warp is rejected."""
    corners = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
    p1 = cv2.perspectiveTransform(corners, h1.astype(np.float64)).reshape(-1, 2)
    p2 = cv2.perspectiveTransform(corners, h2.astype(np.float64)).reshape(-1, 2)
    return float(np.linalg.norm(p1 - p2, axis=1).max()) <= max_frac * max(w, h)


def _erode_valid_mask(valid_mask: Any) -> Any:
    height, width = valid_mask.shape[:2]
    kernel_size = max(7, int(min(height, width) * 0.018))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.erode(valid_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)


def _localized_changes(
    crop: Any,
    origin_x: int,
    origin_y: int,
    image_width: int,
    image_height: int,
    *,
    difference_threshold: int,
    min_mean_delta: float,
) -> list[dict[str, Any]]:
    roi_height, roi_width = crop.shape[:2]
    roi_area = float(max(1, roi_width * roi_height))
    image_area = float(max(1, image_width * image_height))
    _, mask = cv2.threshold(crop, difference_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 3)
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Size the minimum change relative to the (tight, per-part) ROI, not the whole
    # 12 MP frame: image_area*0.000035 ≈ 427px² rejected a ~15px part (~225px²)
    # before it could ever be flagged. The image-area term is kept only as an upper
    # cap so a loose/whole-image ROI never gets a *lower* floor than before (safe:
    # this can only make detection more sensitive, never less).
    min_area = max(18.0, min(roi_area * 0.002, image_area * 0.000035))
    max_area = min(image_area * 0.075, roi_area * 0.22)
    changes: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 or h < 4:
            continue
        bbox_area = float(max(1, w * h))
        if bbox_area > image_area * 0.10 or bbox_area > roi_area * 0.28:
            continue
        fill_ratio = area / bbox_area
        if fill_ratio < 0.12:
            continue
        component_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        values = crop[component_mask > 0]
        if values.size == 0:
            continue
        component_mean = float(values.mean())
        if component_mean < max(min_mean_delta * 2.0, difference_threshold * 0.72):
            continue
        abs_x = origin_x + x
        abs_y = origin_y + y
        abs_contour = contour.copy()
        abs_contour[:, :, 0] += origin_x
        abs_contour[:, :, 1] += origin_y
        changed_fraction = area / image_area
        confidence = _localized_score(component_mean, area, image_area)
        changes.append(
            {
                "region": _bbox_region(abs_x, abs_y, w, h, image_width, image_height),
                "bbox_normalized": _bbox_normalized(abs_x, abs_y, w, h, image_width, image_height),
                "changed_fraction": changed_fraction,
                "mean_delta": component_mean,
                "confidence": confidence,
                "_contour": abs_contour,
            }
        )
    return sorted(changes, key=lambda item: item["confidence"], reverse=True)[:8]


def _localized_score(mean_delta: float, area: float, image_area: float) -> float:
    mean_component = min(mean_delta / 72.0, 1.0)
    size_component = min(area / max(1.0, image_area * 0.004), 1.0)
    return (0.68 * mean_component) + (0.32 * size_component)


def _bbox_normalized(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> dict[str, float]:
    pad = max(3, int(round(max(width, height) * 0.1)))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image_width, x + width + pad)
    bottom = min(image_height, y + height + pad)
    return {
        "x": round(left / image_width, 4),
        "y": round(top / image_height, 4),
        "width": round((right - left) / image_width, 4),
        "height": round((bottom - top) / image_height, 4),
    }


def _bbox_region(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> list[dict[str, float]]:
    bbox = _bbox_normalized(x, y, width, height, image_width, image_height)
    x1 = bbox["x"]
    y1 = bbox["y"]
    x2 = x1 + bbox["width"]
    y2 = y1 + bbox["height"]
    return [
        {"x": round(x1, 4), "y": round(y1, 4)},
        {"x": round(x2, 4), "y": round(y1, 4)},
        {"x": round(x2, 4), "y": round(y2, 4)},
        {"x": round(x1, 4), "y": round(y2, 4)},
    ]


def _dedupe_localized_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    tight_findings = [finding for finding in findings if float(finding.get("_roi_area") or 1.0) <= 0.04]
    pool = tight_findings or findings
    for finding in sorted(pool, key=lambda item: item["confidence"], reverse=True):
        bbox = finding["bbox_normalized"]
        if any(_bbox_iou(bbox, item["bbox_normalized"]) >= 0.28 for item in chosen):
            continue
        chosen.append(finding)
    return chosen[:4]


def _normalized_roi_area(piece: dict[str, Any]) -> float:
    roi = piece.get("roi")
    if not (isinstance(roi, list) and len(roi) == 4):
        return 1.0
    x1, y1, x2, y2 = [float(value) for value in roi]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    left_x2 = left["x"] + left["width"]
    left_y2 = left["y"] + left["height"]
    right_x2 = right["x"] + right["width"]
    right_y2 = right["y"] + right["height"]
    inter_x1 = max(left["x"], right["x"])
    inter_y1 = max(left["y"], right["y"])
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = (left["width"] * left["height"]) + (right["width"] * right["height"]) - intersection
    return intersection / union if union > 0 else 0.0
