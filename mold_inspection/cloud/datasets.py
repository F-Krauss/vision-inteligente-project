from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any
import csv

from mold_inspection.mold_segmenter import normalize_mold_crop, segmentation_mask_png

from .schemas import DatasetFromExamplesRequest, DatasetFromExamplesResponse, new_id
from .storage import ObjectStorage
from .store import MetadataStore

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


def create_dataset_from_examples(
    request: DatasetFromExamplesRequest,
    objects: ObjectStorage,
    store: MetadataStore,
) -> DatasetFromExamplesResponse:
    if not request.ok_image_uris:
        raise ValueError("Dataset requires at least one correct example.")
    if not request.fault_image_uris:
        raise ValueError("Dataset requires at least one incorrect example.")

    dataset_id = new_id("dataset")
    preview_uri = request.ok_image_uris[0]
    normalized_ok_uris = _normalize_examples(request.ok_image_uris, request, objects, "ok")
    normalized_fault_uris = _normalize_examples(request.fault_image_uris, request, objects, "fault")
    rows = _manifest_rows(request, normalized_ok_uris, normalized_fault_uris)
    manifest_bytes = _manifest_bytes(rows)
    mask_bytes = _mask_png_bytes(request, objects, normalized_ok_uris[0])

    manifest_uri = _write_generated_object(
        objects=objects,
        filename="manifest.csv",
        content_type="text/csv",
        family=request.family,
        zone_id=request.zone_id,
        body=manifest_bytes,
    )
    mask_uri = _write_generated_object(
        objects=objects,
        filename="mask.png",
        content_type="image/png",
        family=request.family,
        zone_id=request.zone_id,
        body=mask_bytes,
    )

    response = DatasetFromExamplesResponse(
        id=dataset_id,
        family=request.family,
        zone_id=request.zone_id,
        name=request.name,
        manifest_uri=manifest_uri,
        mask_uri=mask_uri,
        dataset_uri=f"system://{dataset_id}",
        ok_count=len(request.ok_image_uris),
        fault_count=len(request.fault_image_uris),
        piece_count=len(request.expected_pieces),
        preview_image_uri=preview_uri,
    )
    store.put(
        "datasets",
        dataset_id,
        {
            **response.model_dump(),
            "expected_pieces": [piece.model_dump() for piece in request.expected_pieces],
        },
    )
    return response


def _manifest_rows(
    request: DatasetFromExamplesRequest,
    ok_image_uris: list[str],
    fault_image_uris: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, image_uri in enumerate(ok_image_uris, start=1):
        rows.append(_manifest_row(request.family, request.zone_id, image_uri, "ok", index))
    for index, image_uri in enumerate(fault_image_uris, start=1):
        rows.append(_manifest_row(request.family, request.zone_id, image_uri, "fault", index))
    return rows


def _manifest_row(family: str, zone_id: str, image_uri: str, label: str, index: int) -> dict[str, str]:
    split = "train" if label == "ok" and index == 1 else "val"
    return {
        "image_path": image_uri,
        "family": family,
        "zone_id": zone_id,
        "label": label,
        "mold_id": f"{family}_dataset_{label}_{index}",
        "session_id": f"{zone_id}_{label}_{index}",
        "split": split,
    }


def _manifest_bytes(rows: list[dict[str, str]]) -> bytes:
    fields = ["image_path", "family", "zone_id", "label", "mold_id", "session_id", "split"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _normalize_examples(
    image_uris: list[str],
    request: DatasetFromExamplesRequest,
    objects: ObjectStorage,
    label: str,
) -> list[str]:
    normalized_uris: list[str] = []
    for index, image_uri in enumerate(image_uris, start=1):
        source = objects.materialize(image_uri)
        normalized_path = source.parent / f"normalized_{label}_{index}.jpg"
        normalized, segmentation = normalize_mold_crop(source, normalized_path)
        if not segmentation.bbox:
            raise ValueError(f"No se pudo separar el molde del fondo en {label} #{index}.")
        normalized_uris.append(
            _write_generated_object(
                objects=objects,
                filename=normalized.name,
                content_type="image/jpeg",
                family=request.family,
                zone_id=request.zone_id,
                body=normalized.read_bytes(),
            )
        )
    return normalized_uris


def _mask_png_bytes(request: DatasetFromExamplesRequest, objects: ObjectStorage, reference_uri: str) -> bytes:
    if request.mask.type == "png_uri":
        if not request.mask.png_uri:
            raise ValueError("Mask payload with type png_uri requires png_uri.")
        return objects.materialize(request.mask.png_uri).read_bytes()

    if request.mask.type == "auto":
        return segmentation_mask_png(objects.materialize(reference_uri))

    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and numpy are required to generate mask PNG files.")
    if not request.mask.points or len(request.mask.points) < 3:
        raise ValueError("Mask polygon requires at least three points.")

    shape = _reference_shape(reference_uri, objects)
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    points = []
    for point in request.mask.points:
        x = int(round(_normalize_coordinate(point.x) * (width - 1)))
        y = int(round(_normalize_coordinate(point.y) * (height - 1)))
        points.append([x, y])
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise ValueError("Could not encode generated mask.")
    return encoded.tobytes()


def _reference_shape(image_uri: str, objects: ObjectStorage) -> tuple[int, int]:
    image_path = objects.materialize(image_uri)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read reference image for mask: {image_uri}")
    height, width = image.shape[:2]
    return int(height), int(width)


def _normalize_coordinate(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _write_generated_object(
    objects: ObjectStorage,
    filename: str,
    content_type: str,
    family: str,
    zone_id: str,
    body: bytes,
) -> str:
    upload = objects.create_upload(
        filename=filename,
        content_type=content_type,
        family=family,
        zone_id=zone_id,
        purpose="dataset",
    )
    return objects.write_upload(upload.upload_id, body, content_type)


def flat_record(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    if isinstance(data, dict):
        return data
    return record
