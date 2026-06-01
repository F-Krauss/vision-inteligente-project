from __future__ import annotations

from pathlib import Path
from typing import Any


def inspect_expected_pieces(
    family: str,
    zone_id: str,
    image_path: str | Path,
    datasets: list[dict[str, Any]],
    registry_dir: str | Path = "data/model_registry",
    confidence: float = 0.35,
) -> dict[str, Any]:
    expected = _expected_pieces(family, zone_id, datasets)
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
                    "region": piece.get("region"),
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
                    "region": piece.get("region"),
                }
            )

    return {
        "status": "correct" if missing == 0 else "review",
        "message": "Todas las piezas esperadas fueron detectadas." if missing == 0 else "Faltan piezas esperadas.",
        "findings": findings,
        "missing_count": missing,
    }


def _expected_pieces(family: str, zone_id: str, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in reversed(datasets):
        source = record.get("data") if isinstance(record.get("data"), dict) else record
        if source.get("family") == family and source.get("zone_id") == zone_id:
            pieces = source.get("expected_pieces") or []
            if isinstance(pieces, list):
                return [piece for piece in pieces if isinstance(piece, dict)]
    return []
