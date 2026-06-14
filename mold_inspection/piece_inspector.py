from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


def inspect_expected_pieces(
    family: str,
    zone_id: str,
    image_path: str | Path,
    datasets: list[dict[str, Any]],
    registry_dir: str | Path = "data/model_registry",
    confidence: float = 0.35,
    expected_pieces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = expected_pieces or _expected_pieces(family, zone_id, datasets)
    if not expected:
        return {"status": "not_configured", "findings": [], "message": "No hay piezas esperadas configuradas."}

    detector = Path(registry_dir) / family / zone_id / "piece_detector" / "best.pt"
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

    result = YOLO(str(detector)).predict(str(image_path), conf=confidence, verbose=False)[0]
    names = result.names or {}
    detections = []
    for box in result.boxes or []:
        class_id = int(box.cls[0])
        detections.append(
            {
                "class_name": str(names.get(class_id, class_id)),
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )

    findings = []
    missing = 0
    for piece in expected:
        if not piece.get("required", True):
            continue
        match = max(
            (item for item in detections if item["class_name"] == piece["class_name"]),
            key=lambda item: item["confidence"],
            default=None,
        )
        if match:
            findings.append({"piece_id": piece["id"], "class_name": piece["class_name"], "status": "present", **match})
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

    return {
        "status": "correct" if missing == 0 else "review",
        "message": "Todas las piezas esperadas fueron detectadas." if missing == 0 else "Faltan piezas esperadas.",
        "findings": findings,
        "missing_count": missing,
    }


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
    orb = cv2.ORB_create(nfeatures=5000)
    ref_keypoints, ref_desc = orb.detectAndCompute(ref_gray, None)
    cand_keypoints, cand_desc = orb.detectAndCompute(cand_gray, None)
    if ref_desc is None or cand_desc is None:
        alignment["reason"] = "insufficient_features"
        return candidate, alignment

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(ref_desc, cand_desc), key=lambda item: item.distance)
    good_matches = matches[: min(320, len(matches))]
    alignment["matched_keypoints"] = len(good_matches)
    if len(good_matches) < 10:
        alignment["reason"] = "insufficient_matches"
        return candidate, alignment

    ref_points = np.float32([ref_keypoints[item.queryIdx].pt for item in good_matches]).reshape(-1, 1, 2)
    cand_points = np.float32([cand_keypoints[item.trainIdx].pt for item in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(cand_points, ref_points, cv2.RANSAC, 5.0)
    if homography is None or inlier_mask is None:
        alignment["reason"] = "homography_failed"
        return candidate, alignment

    inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))
    alignment["inlier_ratio"] = round(inlier_ratio, 4)
    if inlier_ratio < 0.08:
        alignment["reason"] = "low_inlier_ratio"
        return candidate, alignment

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

    min_area = max(18.0, image_area * 0.000035)
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
