from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


@dataclass(frozen=True)
class MoldSegmentation:
    ok: bool
    confidence: float
    message: str
    guidance: list[str]
    bbox: tuple[int, int, int, int] | None
    bbox_normalized: dict[str, float] | None
    polygon_normalized: list[dict[str, float]]
    mask: Any | None
    image_shape: tuple[int, int]
    mold_area_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "confidence": round(self.confidence, 4),
            "message": self.message,
            "guidance": self.guidance,
            "bbox": list(self.bbox) if self.bbox else None,
            "bbox_normalized": self.bbox_normalized,
            "polygon_normalized": self.polygon_normalized,
            "image_shape": list(self.image_shape),
            "mold_area_ratio": round(self.mold_area_ratio, 4),
        }


def segment_mold(image_path: str | Path, min_confidence: float = 0.35) -> MoldSegmentation:
    if cv2 is None or np is None:
        return MoldSegmentation(
            ok=True,
            confidence=1.0,
            message="Segmentacion generica no disponible; se usa imagen completa.",
            guidance=[],
            bbox=None,
            bbox_normalized=None,
            polygon_normalized=[],
            mask=None,
            image_shape=(0, 0),
            mold_area_ratio=1.0,
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return MoldSegmentation(
            ok=False,
            confidence=0.0,
            message="No se pudo segmentar el molde.",
            guidance=["Vuelve a tomar la foto; no se pudo leer la imagen."],
            bbox=None,
            bbox_normalized=None,
            polygon_normalized=[],
            mask=None,
            image_shape=(0, 0),
        )
    neural = _segment_with_trained_model(Path(image_path), image, min_confidence)
    return neural or segment_mold_image(image, min_confidence=min_confidence)


def segment_mold_image(image, min_confidence: float = 0.35) -> MoldSegmentation:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 35, 115)
    kernel = np.ones((7, 7), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * 0.02
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return MoldSegmentation(
            ok=False,
            confidence=0.0,
            message="No se detecto el molde contra el fondo.",
            guidance=["Centra el molde y evita fondos con poco contraste."],
            bbox=None,
            bbox_normalized=None,
            polygon_normalized=[],
            mask=np.zeros((height, width), dtype=np.uint8),
            image_shape=(height, width),
        )

    contour = max(contours, key=cv2.contourArea)
    x, y, box_width, box_height = cv2.boundingRect(contour)
    pad_x = max(8, int(box_width * 0.06))
    pad_y = max(8, int(box_height * 0.06))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + box_width + pad_x)
    y2 = min(height, y + box_height + pad_y)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    polygon = _normalized_polygon(contour, width, height)
    mold_area_ratio = float(np.count_nonzero(mask)) / float(width * height)
    center_x = (x1 + x2) / 2.0 / width
    center_y = (y1 + y2) / 2.0 / height
    box_area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
    confidence = max(0.0, min(1.0, (mold_area_ratio / 0.18) * 0.55 + min(box_area_ratio, 0.75) * 0.6))
    guidance = _guidance(center_x, center_y, x2 - x1, y2 - y1, width, height)
    if confidence < min_confidence:
        guidance.append("Mejora el contraste entre molde y fondo.")
    return MoldSegmentation(
        ok=confidence >= min_confidence and not guidance,
        confidence=confidence,
        message="Molde detectado contra el fondo." if confidence >= min_confidence else "Segmentacion de molde con baja confianza.",
        guidance=guidance,
        bbox=(x1, y1, x2, y2),
        bbox_normalized={
            "x": round(x1 / width, 4),
            "y": round(y1 / height, 4),
            "width": round((x2 - x1) / width, 4),
            "height": round((y2 - y1) / height, 4),
        },
        polygon_normalized=polygon,
        mask=mask,
        image_shape=(height, width),
        mold_area_ratio=mold_area_ratio,
    )


def normalize_mold_crop(image_path: str | Path, output_path: str | Path, image_size: int = 768) -> tuple[Path, MoldSegmentation]:
    if cv2 is None or np is None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        image = Path(image_path).read_bytes()
        output.write_bytes(image)
        return output, segment_mold(image_path)

    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Could not read image: {image_path}")
    segmentation = segment_mold_image(source)
    if not segmentation.bbox:
        raise ValueError(segmentation.message)
    x1, y1, x2, y2 = segmentation.bbox
    crop = source[y1:y2, x1:x2]
    normalized = _letterbox(crop, image_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), normalized)
    return output, segmentation


def segmentation_mask_png(image_path: str | Path, image_size: int = 768) -> bytes:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy are required to generate mold masks.")
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Could not read image: {image_path}")
    segmentation = segment_mold_image(source)
    if not segmentation.bbox or segmentation.mask is None:
        raise ValueError(segmentation.message)
    x1, y1, x2, y2 = segmentation.bbox
    crop_mask = segmentation.mask[y1:y2, x1:x2]
    normalized_mask = _letterbox(crop_mask, image_size, grayscale=True)
    _, binary = cv2.threshold(normalized_mask, 127, 255, cv2.THRESH_BINARY)
    ok, encoded = cv2.imencode(".png", binary)
    if not ok:
        raise ValueError("Could not encode generated mold mask.")
    return encoded.tobytes()


def _letterbox(image, size: int, grayscale: bool = False):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_NEAREST if grayscale else cv2.INTER_AREA
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    if grayscale:
        canvas = np.zeros((size, size), dtype=np.uint8)
    else:
        canvas = np.full((size, size, 3), 0, dtype=np.uint8)
    x = (size - resized_width) // 2
    y = (size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _guidance(center_x: float, center_y: float, box_width: int, box_height: int, width: int, height: int) -> list[str]:
    guidance: list[str] = []
    width_ratio = box_width / width
    height_ratio = box_height / height
    if width_ratio < 0.42 and height_ratio < 0.42:
        guidance.append("Acércate al molde.")
    elif width_ratio > 0.94 or height_ratio > 0.94:
        guidance.append("Aléjate un poco.")
    if center_x < 0.42:
        guidance.append("Mueve a la derecha.")
    elif center_x > 0.58:
        guidance.append("Mueve a la izquierda.")
    if center_y < 0.42:
        guidance.append("Baja la cámara.")
    elif center_y > 0.58:
        guidance.append("Sube la cámara.")
    return guidance


def _segment_with_trained_model(image_path: Path, image, min_confidence: float) -> MoldSegmentation | None:
    model_path = Path(os.getenv("MOLD_SEGMENTER_MODEL", "data/segmenter/best.pt"))
    if not model_path.exists():
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    height, width = image.shape[:2]
    try:
        result = YOLO(str(model_path)).predict(str(image_path), conf=min_confidence, verbose=False)[0]
    except Exception:
        return None
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return MoldSegmentation(
            ok=False,
            confidence=0.0,
            message="La red segmentadora no detecto el molde.",
            guidance=["Centra el molde y toma otra foto."],
            bbox=None,
            bbox_normalized=None,
            polygon_normalized=[],
            mask=np.zeros((height, width), dtype=np.uint8),
            image_shape=(height, width),
        )

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    index = int(areas.argmax())
    x1, y1, x2, y2 = [int(round(value)) for value in boxes[index]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    mask = (result.masks.data[index].cpu().numpy() > 0.5).astype("uint8") * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    polygon = [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in result.masks.xyn[index]]
    center_x = (x1 + x2) / 2.0 / width
    center_y = (y1 + y2) / 2.0 / height
    guidance = _guidance(center_x, center_y, x2 - x1, y2 - y1, width, height)
    confidence = float(confidences[index])
    if confidence < min_confidence:
        guidance.append("Segmentacion de molde con baja confianza.")
    return MoldSegmentation(
        ok=confidence >= min_confidence and not guidance,
        confidence=confidence,
        message="Molde detectado por red segmentadora." if confidence >= min_confidence else "Red segmentadora con baja confianza.",
        guidance=guidance,
        bbox=(x1, y1, x2, y2),
        bbox_normalized={
            "x": round(x1 / width, 4),
            "y": round(y1 / height, 4),
            "width": round((x2 - x1) / width, 4),
            "height": round((y2 - y1) / height, 4),
        },
        polygon_normalized=polygon,
        mask=mask,
        image_shape=(height, width),
        mold_area_ratio=float(np.count_nonzero(mask)) / float(width * height),
    )


def _normalized_polygon(contour, width: int, height: int) -> list[dict[str, float]]:
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.012 * perimeter, True)
    points = approximation.reshape(-1, 2)
    if len(points) < 3:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        points = np.array(
            [
                [x, y],
                [x + box_width, y],
                [x + box_width, y + box_height],
                [x, y + box_height],
            ]
        )
    return [
        {
            "x": round(max(0.0, min(1.0, float(x) / width)), 4),
            "y": round(max(0.0, min(1.0, float(y) / height)), 4),
        }
        for x, y in points[:24]
    ]
