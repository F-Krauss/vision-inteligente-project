from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import re
import shutil

from .config import CloudSettings
from .model_artifacts import materialize_model_uri
from .references import object_public_url
from .references import expected_pieces_for_zone
from .schemas import (
    AnnotationCreateRequest,
    AnnotationRecord,
    AnnotationTransferRequest,
    AnnotationTransferResponse,
    AnnotationTransferResult,
    AutoAnnotationDraftRequest,
    AutoAnnotationDraftResponse,
    DatasetFromAnnotationsRequest,
    DatasetFromAnnotationsResponse,
    PieceAnnotationPayload,
    utc_now,
)
from .storage import ObjectStorage
from .store import MetadataStore

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


MANIFEST_FIELDS = ["image_path", "family", "zone_id", "label", "mold_id", "session_id", "split", "annotation_id", "image_uri"]


def save_annotation(request: AnnotationCreateRequest, store: MetadataStore) -> AnnotationRecord:
    if not request.annotations:
        raise ValueError("Annotation payload requires at least one box.")
    normalized = [_normalize_annotation(item, index) for index, item in enumerate(request.annotations, start=1)]
    image_id = request.image_id or _image_id_from_uri(request.image_uri)
    record_id = _annotation_record_id(request.family, request.zone_id, image_id)
    existing = store.get("annotations", record_id) or {}
    record = AnnotationRecord(
        id=record_id,
        created_at=str(existing.get("created_at") or utc_now()),
        image_id=image_id,
        image_uri=request.image_uri,
        image_url=object_public_url(request.image_uri),
        family=request.family,
        zone_id=request.zone_id,
        mold_id=request.mold_id,
        session_id=request.session_id,
        operator_id=request.operator_id,
        reference_id=request.reference_id,
        split=request.split,
        annotations=normalized,
        box_count=len(normalized),
        metadata=request.metadata,
    )
    store.put("annotations", record_id, record.model_dump())
    return record


def transfer_annotations(request: AnnotationTransferRequest, objects: ObjectStorage) -> AnnotationTransferResponse:
    """Map the reference annotations onto each comparison image via homography."""
    from mold_inspection.piece_inspector import transfer_annotations as _warp_onto_targets

    if not request.annotations:
        raise ValueError("No hay anotaciones de referencia para mapear.")
    if not request.target_image_uris:
        raise ValueError("Faltan imágenes de comparación.")

    reference_path = objects.materialize(request.reference_image_uri)
    target_paths = [objects.materialize(uri) for uri in request.target_image_uris]
    source = [item.model_dump() for item in request.annotations]
    raw_results = _warp_onto_targets(reference_path, target_paths, source)

    results: list[AnnotationTransferResult] = []
    for uri, item in zip(request.target_image_uris, raw_results):
        results.append(
            AnnotationTransferResult(
                image_uri=uri,
                ok=bool(item.get("ok")),
                alignment_confidence=float(item.get("confidence") or 0.0),
                message=str(item.get("message") or ""),
                annotations=[PieceAnnotationPayload(**ann) for ann in item.get("annotations", [])],
            )
        )
    return AnnotationTransferResponse(results=results)


def list_annotations(
    store: MetadataStore,
    family: str | None = None,
    zone_id: str | None = None,
    image_id: str | None = None,
    image_uri: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in store.list("annotations"):
        source = _flat(record)
        if family and source.get("family") != family:
            continue
        if zone_id and source.get("zone_id") != zone_id:
            continue
        if image_id and source.get("image_id") != image_id:
            continue
        if image_uri and source.get("image_uri") != image_uri:
            continue
        records.append(_with_public_url(source))
    return records


def create_auto_annotation_draft(
    request: AutoAnnotationDraftRequest,
    objects: ObjectStorage,
    store: MetadataStore,
) -> AutoAnnotationDraftResponse:
    model_result = _draft_from_model(request, objects, store)
    if model_result:
        return model_result

    template_result = _draft_from_latest_annotation(request, store)
    if template_result:
        return template_result

    roi_boxes = _drafts_from_expected_rois(request, store)
    if roi_boxes:
        return AutoAnnotationDraftResponse(
            family=request.family,
            zone_id=request.zone_id,
            image_uri=request.image_uri,
            source="roi",
            annotations=roi_boxes,
            message="Borrador generado desde ROIs configuradas. Corrige antes de guardar.",
        )

    return AutoAnnotationDraftResponse(
        family=request.family,
        zone_id=request.zone_id,
        image_uri=request.image_uri,
        source="empty",
        annotations=[],
        message="No hay modelo, anotaciones previas ni ROIs para auto-anotar esta zona.",
    )


def create_dataset_from_annotations(
    request: DatasetFromAnnotationsRequest,
    settings: CloudSettings,
    objects: ObjectStorage,
    store: MetadataStore,
) -> DatasetFromAnnotationsResponse:
    annotations = [
        _flat(record)
        for record in store.list("annotations")
        if _matches_dataset_request(_flat(record), request)
    ]
    if request.annotation_ids:
        wanted = set(request.annotation_ids)
        annotations = [record for record in annotations if str(record.get("id")) in wanted]
    if not annotations:
        raise ValueError("No hay anotaciones para esta familia y zona.")

    annotations = _ensure_validation_split(annotations)
    class_names = _class_names(annotations)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    response = DatasetFromAnnotationsResponse(
        family=request.family,
        zone_id=request.zone_id,
        name=request.name,
        dataset_uri="",
        data_yaml_uri="",
        manifest_uri="",
        mask_uri="",
        image_count=len(annotations),
        box_count=sum(len(record.get("annotations") or []) for record in annotations),
        class_count=len(class_names),
        train_count=sum(1 for record in annotations if record.get("split") == "train"),
        val_count=sum(1 for record in annotations if record.get("split") == "val"),
        test_count=sum(1 for record in annotations if record.get("split") == "test"),
        preview_image_uri=str(annotations[0].get("image_uri") or ""),
    )
    root = settings.local_state_dir / "annotation_datasets" / response.id
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    first_image_path: Path | None = None
    for index, annotation in enumerate(annotations, start=1):
        source = objects.materialize(str(annotation["image_uri"]))
        if first_image_path is None:
            first_image_path = source
        split = str(annotation.get("split") or "train")
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        image_name = f"{index:06d}{suffix}"
        image_out = root / "images" / split / image_name
        shutil.copy2(source, image_out)
        label_out = root / "labels" / split / f"{Path(image_name).stem}.txt"
        label_out.write_text("\n".join(_yolo_lines(annotation.get("annotations") or [], class_to_id)) + "\n")
        rows.append(
            {
                "image_path": image_out.relative_to(root).as_posix(),
                "family": request.family,
                "zone_id": request.zone_id,
                "label": "annotated",
                "mold_id": str(annotation.get("mold_id") or ""),
                "session_id": str(annotation.get("session_id") or ""),
                "split": split,
                "annotation_id": str(annotation.get("id") or ""),
                "image_uri": str(annotation.get("image_uri") or ""),
            }
        )

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    data_yaml = root / "data.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_block}\n"
    )
    (root / "classes.json").write_text(json.dumps(class_to_id, indent=2) + "\n")
    mask = root / "mask.png"
    _write_full_mask(first_image_path, mask)

    if settings.artifact_bucket:
        dataset_prefix = f"annotation_datasets/{response.id}"
        _upload_tree_to_gcs(root, settings.artifact_bucket, dataset_prefix)
        response.dataset_uri = f"gs://{settings.artifact_bucket}/{dataset_prefix}"
        response.data_yaml_uri = f"{response.dataset_uri}/data.yaml"
        response.manifest_uri = f"{response.dataset_uri}/manifest.csv"
        response.mask_uri = f"{response.dataset_uri}/mask.png"
    else:
        response.dataset_uri = f"file://{root}"
        response.data_yaml_uri = f"file://{data_yaml}"
        response.manifest_uri = f"file://{manifest}"
        response.mask_uri = f"file://{mask}"
    payload = response.model_dump()
    payload["expected_pieces"] = _expected_from_annotations(annotations)
    store.put("annotation_datasets", response.id, payload)
    store.put("datasets", response.id, payload)
    return response


def _normalize_annotation(item: PieceAnnotationPayload, index: int) -> PieceAnnotationPayload:
    if len(item.bbox) != 4:
        raise ValueError(f"La caja #{index} debe tener [x1, y1, x2, y2].")
    x1, y1, x2, y2 = [_clip(value) for value in item.bbox]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"La caja #{index} no tiene tamaño válido.")
    polygon = None
    if item.polygon:
        polygon = [[_clip(p[0]), _clip(p[1])] for p in item.polygon if len(p) == 2]
    return PieceAnnotationPayload(
        id=item.id or f"box_{index:03d}",
        element_id=item.element_id or item.id or f"box_{index:03d}",
        class_name=item.class_name,
        bbox=[x1, y1, x2, y2],
        status=item.status,
        notes=item.notes,
        shape=item.shape,
        polygon=polygon,
        category_id=item.category_id,
        category_name=item.category_name,
        importance=item.importance,
    )


def _draft_from_model(
    request: AutoAnnotationDraftRequest,
    objects: ObjectStorage,
    store: MetadataStore,
) -> AutoAnnotationDraftResponse | None:
    model_version = _resolve_model_version(request, store)
    model_uri = str(model_version.get("model_uri") or "") if model_version else ""
    if not model_uri:
        return None
    settings = getattr(objects, "settings", None)
    if isinstance(settings, CloudSettings):
        model_path = materialize_model_uri(model_uri, settings)
    else:
        model_path = Path(model_uri.removeprefix("file://")) if model_uri.startswith("file://") else Path(model_uri)
    if not model_path or not model_path.exists():
        return None
    if model_path.suffix.lower() == ".json":
        return _draft_from_template_model(request, model_version, model_path)
    try:
        from mold_inspection.piece_inspector import PIECE_DETECTOR_IMGSZ, predict_piece_detections
    except (ImportError, RuntimeError):
        return None

    image_path = objects.materialize(request.image_uri)
    try:
        detections, width, height = predict_piece_detections(
            model_path,
            image_path,
            confidence=request.confidence,
            imgsz=PIECE_DETECTOR_IMGSZ,
            tile=True,
        )
    except RuntimeError:
        return None
    drafts: list[PieceAnnotationPayload] = []
    if width <= 0 or height <= 0:
        return None
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
        class_name = str(detection["class_name"])
        drafts.append(
            PieceAnnotationPayload(
                id=f"auto_model_{index:03d}",
                element_id=class_name,
                class_name=class_name,
                bbox=[_clip(x1 / width), _clip(y1 / height), _clip(x2 / width), _clip(y2 / height)],
                status="present",
                notes=f"auto_draft_model:{model_version.get('id')}",
            )
        )
    if not drafts:
        return None
    return AutoAnnotationDraftResponse(
        family=request.family,
        zone_id=request.zone_id,
        image_uri=request.image_uri,
        source="model",
        model_version_id=str(model_version.get("id") or ""),
        annotations=drafts,
        message="Borrador generado con modelo promovido. Corrige antes de guardar.",
    )


def _draft_from_template_model(
    request: AutoAnnotationDraftRequest,
    model_version: dict[str, Any],
    model_path: Path,
) -> AutoAnnotationDraftResponse | None:
    try:
        payload = json.loads(model_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("type") != "annotation_template_detector":
        return None
    if payload.get("family") != request.family or payload.get("zone_id") != request.zone_id:
        return None
    drafts: list[PieceAnnotationPayload] = []
    for index, item in enumerate(payload.get("boxes") or [], start=1):
        draft = _draft_from_annotation_item(
            item,
            draft_id=f"auto_model_{index:03d}",
            note=f"auto_draft_model:{model_version.get('id')}",
        )
        if draft:
            drafts.append(draft)
    if not drafts:
        return None
    return AutoAnnotationDraftResponse(
        family=request.family,
        zone_id=request.zone_id,
        image_uri=request.image_uri,
        source="model",
        model_version_id=str(model_version.get("id") or ""),
        annotations=drafts,
        message="Borrador generado con artifact local del modelo promovido. Corrige antes de guardar.",
    )


def _draft_from_latest_annotation(
    request: AutoAnnotationDraftRequest,
    store: MetadataStore,
) -> AutoAnnotationDraftResponse | None:
    templates = [
        _flat(record)
        for record in store.list("annotations")
        if _flat(record).get("family") == request.family and _flat(record).get("zone_id") == request.zone_id
    ]
    templates = [record for record in templates if record.get("image_uri") != request.image_uri and record.get("annotations")]
    if not templates:
        return None
    template = templates[-1]
    drafts: list[PieceAnnotationPayload] = []
    for index, item in enumerate(template.get("annotations") or [], start=1):
        draft = _draft_from_annotation_item(
            item,
            draft_id=f"auto_template_{index:03d}",
            note=f"auto_draft_template:{template.get('id')}",
        )
        if draft:
            drafts.append(draft)
    if not drafts:
        return None
    return AutoAnnotationDraftResponse(
        family=request.family,
        zone_id=request.zone_id,
        image_uri=request.image_uri,
        source="annotation_template",
        annotations=drafts,
        message="Borrador generado desde la última anotación de esta zona. Corrige antes de guardar.",
    )


def _drafts_from_expected_rois(request: AutoAnnotationDraftRequest, store: MetadataStore) -> list[PieceAnnotationPayload]:
    drafts: list[PieceAnnotationPayload] = []
    for index, piece in enumerate(expected_pieces_for_zone(request.zone_id, store, family=request.family), start=1):
        roi = piece.get("roi")
        if not isinstance(roi, list) or len(roi) != 4:
            continue
        class_name = str(piece.get("class_name") or piece.get("id") or "piece")
        drafts.append(
            PieceAnnotationPayload(
                id=f"auto_roi_{index:03d}",
                element_id=str(piece.get("id") or class_name),
                class_name=class_name,
                bbox=[_clip(value) for value in roi],
                status="present",
                notes="auto_draft_roi",
            )
        )
    return drafts


def _draft_from_annotation_item(item: dict[str, Any], draft_id: str, note: str) -> PieceAnnotationPayload | None:
    if item.get("status", "present") not in {"present", "uncertain"}:
        return None
    bbox = item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    polygon = item.get("polygon")
    normalized_polygon = None
    if isinstance(polygon, list):
        normalized_polygon = [
            [_clip(point[0]), _clip(point[1])]
            for point in polygon
            if isinstance(point, list) and len(point) == 2
        ]
        if len(normalized_polygon) < 3:
            normalized_polygon = None
    class_name = str(item.get("class_name") or "piece")
    return PieceAnnotationPayload(
        id=draft_id,
        element_id=str(item.get("element_id") or item.get("id") or class_name),
        class_name=class_name,
        bbox=[_clip(value) for value in bbox],
        status="present",
        notes=note,
        shape="rect" if item.get("shape") == "rect" else "polygon",
        polygon=normalized_polygon,
        category_id=item.get("category_id"),
        category_name=item.get("category_name"),
        importance=item.get("importance") if item.get("importance") in {"critical", "relevant", "minor"} else None,
    )


def _resolve_model_version(request: AutoAnnotationDraftRequest, store: MetadataStore) -> dict[str, Any] | None:
    if request.model_version_id:
        return store.get("model_versions", request.model_version_id)
    model_id = f"best_{request.family}_{request.zone_id}".replace("/", "_")
    return store.get("model_versions", model_id)


def _matches_dataset_request(record: dict[str, Any], request: DatasetFromAnnotationsRequest) -> bool:
    return record.get("family") == request.family and record.get("zone_id") == request.zone_id


def _ensure_validation_split(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(record.get("split") == "val" for record in records) or len(records) < 2:
        return records
    copied = [dict(record) for record in records]
    copied[-1]["split"] = "val"
    return copied


def _class_names(records: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for record in records:
        for item in record.get("annotations") or []:
            class_name = str(item.get("class_name") or "piece")
            if class_name not in names:
                names.append(class_name)
    return names or ["piece"]


def _upload_tree_to_gcs(root: Path, bucket_name: str, prefix: str) -> None:
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-storage is required to upload annotation datasets") from exc
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        blob_name = f"{prefix.rstrip('/')}/{path.relative_to(root).as_posix()}"
        bucket.blob(blob_name).upload_from_filename(path)


def _yolo_lines(items: list[dict[str, Any]], class_to_id: dict[str, int]) -> list[str]:
    lines: list[str] = []
    for item in items:
        if item.get("status", "present") != "present":
            continue
        class_name = str(item.get("class_name") or "piece")
        if class_name not in class_to_id:
            continue
        x1, y1, x2, y2 = [_clip(value) for value in item.get("bbox", [0, 0, 0, 0])]
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        lines.append(f"{class_to_id[class_name]} {(x1 + width / 2):.6f} {(y1 + height / 2):.6f} {width:.6f} {height:.6f}")
    return lines


def _write_full_mask(image_path: Path | None, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is None or np is None or image_path is None:
        output_path.write_bytes(
            bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108000000003a7e9b550000000a49444154789c6360000000020001e221bc330000000049454e44ae426082")
        )
        return
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        output_path.write_bytes(
            bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108000000003a7e9b550000000a49444154789c6360000000020001e221bc330000000049454e44ae426082")
        )
        return
    height, width = image.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    cv2.imwrite(str(output_path), mask)


def _expected_from_classes(class_names: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": name,
            "name": name.replace("_", " ").title(),
            "class_name": name,
            "required": True,
            "critical": True,
        }
        for name in class_names
    ]


def _expected_from_annotations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pieces: dict[str, dict[str, Any]] = {}
    for record in records:
        for index, item in enumerate(record.get("annotations") or [], start=1):
            if item.get("status", "present") not in {"present", "uncertain"}:
                continue
            bbox = item.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            class_name = str(item.get("class_name") or "piece")
            piece_id = str(item.get("element_id") or item.get("id") or f"{class_name}_{index:03d}")
            pieces.setdefault(
                piece_id,
                {
                    "id": piece_id,
                    "name": str(item.get("category_name") or class_name).replace("_", " ").title(),
                    "class_name": class_name,
                    "roi": [_clip(value) for value in bbox],
                    "required": True,
                    "critical": item.get("importance") != "minor",
                },
            )
    return list(pieces.values()) or _expected_from_classes(_class_names(records))


def _with_public_url(record: dict[str, Any]) -> dict[str, Any]:
    next_record = dict(record)
    if next_record.get("image_uri"):
        next_record["image_url"] = object_public_url(str(next_record["image_uri"]))
    return next_record


def _annotation_record_id(family: str, zone_id: str, image_id: str) -> str:
    return f"ann_{_safe_token(family)}_{_safe_token(zone_id)}_{_safe_token(image_id)}"


def _image_id_from_uri(image_uri: str) -> str:
    if image_uri.startswith("local://"):
        return image_uri.removeprefix("local://")
    return Path(image_uri.removeprefix("file://")).stem or _safe_token(image_uri)[-24:]


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "image"


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _flat(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    return data if isinstance(data, dict) else record
