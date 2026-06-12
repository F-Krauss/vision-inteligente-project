from __future__ import annotations

from pathlib import Path
from typing import Any

from mold_inspection.mold_segmenter import segment_mold

from .pipeline import evaluate_capture_quality
from .references import get_zone_reference
from .schemas import CaptureGuidanceRequest, CaptureGuidanceResponse
from .storage import ObjectStorage
from .store import MetadataStore

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


def create_capture_guidance(
    request: CaptureGuidanceRequest,
    objects: ObjectStorage,
    store: MetadataStore,
    model_registry_dir: Path,
) -> CaptureGuidanceResponse:
    image_path = objects.materialize(request.image_uri)
    quality = evaluate_capture_quality(image_path)
    if not quality["ok"]:
        return CaptureGuidanceResponse(
            ok=False,
            message=quality["message"],
            guidance=quality["guidance"],
            quality=quality,
            alignment={},
        )

    reference_path = _reference_path(request, objects, store, model_registry_dir)
    alignment = _evaluate_alignment(image_path, reference_path)
    guidance = list(alignment.pop("guidance", []))
    return CaptureGuidanceResponse(
        ok=not guidance,
        auto_capture_ready=not guidance and bool(alignment.get("auto_capture_ready")),
        message="Encuadre correcto. Mantén estable y captura." if not guidance else guidance[0],
        guidance=guidance or ["Mantén estable y captura."],
        quality=quality,
        alignment=alignment,
    )


def _evaluate_alignment(image_path: Path, reference_path: Path | None = None) -> dict[str, Any]:
    current_segmentation = segment_mold(image_path, min_confidence=0.25)
    current = _metrics_from_segmentation(current_segmentation)
    if not current:
        return {
            "ok": False,
            "guidance": current_segmentation.guidance or ["Centra el molde dentro del marco."],
            "object_area_ratio": 0.0,
            "mold_segmentation": current_segmentation.to_dict(),
        }

    reference = None
    if reference_path:
        reference = _metrics_from_segmentation(segment_mold(reference_path, min_confidence=0.1))

    target_center_x = reference["object_center_x"] if reference else 0.5
    target_center_y = reference["object_center_y"] if reference else 0.5
    target_width = reference["object_width_ratio"] if reference else 0.64
    target_height = reference["object_height_ratio"] if reference else 0.64
    guidance: list[str] = []

    if current["object_width_ratio"] < target_width * 0.78 and current["object_height_ratio"] < target_height * 0.78:
        guidance.append("Acércate al molde.")
    elif current["object_width_ratio"] > target_width * 1.18 or current["object_height_ratio"] > target_height * 1.18:
        guidance.append("Aléjate un poco.")

    if current["object_center_x"] < target_center_x - 0.08:
        guidance.append("Mueve a la derecha.")
    elif current["object_center_x"] > target_center_x + 0.08:
        guidance.append("Mueve a la izquierda.")
    if current["object_center_y"] < target_center_y - 0.08:
        guidance.append("Baja la cámara.")
    elif current["object_center_y"] > target_center_y + 0.08:
        guidance.append("Sube la cámara.")

    return {
        "ok": not guidance,
        "guidance": guidance,
        **{key: round(value, 4) for key, value in current.items()},
        "mold_segmentation": current_segmentation.to_dict(),
        "auto_capture_ready": not guidance and current_segmentation.confidence >= 0.65,
        "pose_score": _pose_score(current, target_center_x, target_center_y, target_width, target_height),
        "reference": str(reference_path) if reference_path else None,
        "target_center_x": round(target_center_x, 4),
        "target_center_y": round(target_center_y, 4),
        "target_width_ratio": round(target_width, 4),
        "target_height_ratio": round(target_height, 4),
    }


def _pose_score(current: dict[str, float], target_center_x: float, target_center_y: float, target_width: float, target_height: float) -> float:
    center_error = abs(current["object_center_x"] - target_center_x) + abs(current["object_center_y"] - target_center_y)
    scale_error = abs(current["object_width_ratio"] - target_width) + abs(current["object_height_ratio"] - target_height)
    return round(max(0.0, min(1.0, 1.0 - center_error * 2.2 - scale_error * 0.9)), 4)


def _metrics_from_segmentation(segmentation) -> dict[str, float] | None:
    bbox = segmentation.bbox_normalized
    if not bbox:
        return None
    return {
        "object_center_x": float(bbox["x"]) + float(bbox["width"]) / 2.0,
        "object_center_y": float(bbox["y"]) + float(bbox["height"]) / 2.0,
        "object_width_ratio": float(bbox["width"]),
        "object_height_ratio": float(bbox["height"]),
        "object_area_ratio": float(segmentation.mold_area_ratio),
    }


def _object_metrics(image) -> dict[str, float] | None:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 110)
    kernel = np.ones((7, 7), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * 0.02
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(contour)
    return {
        "object_center_x": (x + box_width / 2.0) / width,
        "object_center_y": (y + box_height / 2.0) / height,
        "object_width_ratio": box_width / width,
        "object_height_ratio": box_height / height,
        "object_area_ratio": (box_width * box_height) / float(width * height),
    }


def _reference_path(
    request: CaptureGuidanceRequest,
    objects: ObjectStorage,
    store: MetadataStore,
    model_registry_dir: Path,
) -> Path | None:
    zone_reference = get_zone_reference(
        request.zone_id,
        store,
        family=request.family,
        reference_id=request.reference_id,
    )
    if zone_reference and zone_reference.get("image_uri"):
        try:
            return objects.materialize(str(zone_reference["image_uri"]))
        except ValueError:
            pass
    model_anchor = model_registry_dir / request.family / request.zone_id / "best_model" / "anchor.jpg"
    if model_anchor.exists():
        return model_anchor
    for record in reversed(store.list("datasets")):
        source = record.get("data") if isinstance(record.get("data"), dict) else record
        if source.get("family") == request.family and source.get("zone_id") == request.zone_id:
            preview = source.get("preview_image_uri")
            if not preview:
                return None
            try:
                return objects.materialize(str(preview))
            except ValueError:
                return None
    return None
