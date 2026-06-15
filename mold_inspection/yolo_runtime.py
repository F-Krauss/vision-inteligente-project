from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

from .decision import inspect_zone
from .models import Box, Detection, InspectionConfig


# Defaults tuned for tiny parts (~15px on ~4032px frames): a larger backbone than
# nano and a 1280 training size that matches inference (see piece_inspector
# PIECE_DETECTOR_IMGSZ). Override per zone via the train CLI / training request.
def train_yolo(
    data_yaml: str | Path,
    weights: str = "yolo11s.pt",
    epochs: int = 80,
    image_size: int = 1280,
    project: str | Path | None = None,
    name: str = "piece_detector",
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install vision dependencies: python -m pip install -e ".[vision]"') from exc

    model = YOLO(weights)
    kwargs = {"data": str(data_yaml), "epochs": epochs, "imgsz": image_size}
    if project is not None:
        kwargs.update({"project": str(project), "name": name, "exist_ok": True})
    model.train(**kwargs)


def inspect_images(
    weights: str | Path,
    config_path: str | Path,
    family: str,
    zone_id: str,
    images: Iterable[str | Path],
    confidence: float = 0.25,
) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install vision dependencies: python -m pip install -e ".[vision]"') from exc

    config = InspectionConfig.load(config_path)
    zone = config.zone(family, zone_id)
    model = YOLO(str(weights))

    reports: list[dict] = []
    for image_path in images:
        image_path = Path(image_path)
        prediction = model.predict(str(image_path), conf=confidence, verbose=False)[0]
        detections = _detections_from_prediction(prediction)
        result = inspect_zone(zone, detections)
        reports.append(
            {
                "image_path": str(image_path),
                "family": family,
                "zone_id": zone_id,
                "result": result.as_dict(),
            }
        )
    return reports


def write_report(path: str | Path, reports: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reports": reports}, indent=2) + "\n")


def _detections_from_prediction(prediction) -> list[Detection]:
    names = prediction.names
    boxes = prediction.boxes
    width = float(prediction.orig_shape[1])
    height = float(prediction.orig_shape[0])
    detections: list[Detection] = []

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            Detection(
                class_name=str(names[class_id]),
                confidence=confidence,
                bbox=Box(x1 / width, y1 / height, x2 / width, y2 / height),
            )
        )
    return detections
