from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from .schemas import ExpectedPieceRecord, ReferenceCreateRequest, ReferenceRecord, utc_now
from .storage import ObjectStorage
from .store import MetadataStore

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


DEFAULT_TOLERANCE = {
    "translation": 0.08,
    "scale": 0.2,
    "rotation": 8.0,
    "pose_score": 0.72,
}


def create_zone_reference(
    zone_id: str,
    request: ReferenceCreateRequest,
    objects: ObjectStorage,
    store: MetadataStore,
) -> ReferenceRecord:
    image_path = objects.materialize(request.image_uri)
    width, height = _image_size(image_path)
    reference_id = _safe_token(request.reference_id or "default")
    record_id = _reference_record_id(request.family, zone_id, reference_id)
    tolerance = {**DEFAULT_TOLERANCE, **request.tolerance}
    record = ReferenceRecord(
        id=record_id,
        family=request.family,
        zone_id=zone_id,
        reference_id=reference_id,
        image_uri=request.image_uri,
        image_url=object_public_url(request.image_uri),
        mask_uri=request.mask_uri,
        mask_url=object_public_url(request.mask_uri) if request.mask_uri else None,
        width=width,
        height=height,
        tolerance=tolerance,
    )
    payload = record.model_dump()
    store.put("references", record_id, payload)
    _upsert_zone_reference(zone_id, payload, store)
    return record


def get_zone_reference(
    zone_id: str,
    store: MetadataStore,
    family: str | None = None,
    reference_id: str | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for record in store.list("references"):
        source = _flat(record)
        if source.get("zone_id") != zone_id:
            continue
        if family and source.get("family") != family:
            continue
        if reference_id and source.get("reference_id") != reference_id:
            continue
        candidates.append(source)
    if candidates:
        chosen = sorted(candidates, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""))[-1]
        return _with_public_urls(chosen)

    for record in store.list("zones"):
        source = _flat(record)
        if source.get("id") != zone_id and source.get("zone_id") != zone_id:
            continue
        if family and source.get("family") not in {family, None, ""}:
            continue
        image_uri = source.get("reference_image_uri") or source.get("image_uri")
        if not image_uri:
            continue
        return _with_public_urls(
            {
                "id": _reference_record_id(str(source.get("family") or family or "default"), zone_id, str(source.get("reference_id") or "default")),
                "family": str(source.get("family") or family or "default"),
                "zone_id": zone_id,
                "reference_id": str(source.get("reference_id") or "default"),
                "image_uri": str(image_uri),
                "mask_uri": source.get("mask_uri"),
                "tolerance": source.get("tolerance") or DEFAULT_TOLERANCE,
            }
        )
    return None


def expected_pieces_for_zone(zone_id: str, store: MetadataStore, family: str | None = None) -> list[dict[str, Any]]:
    dataset_pieces: list[dict[str, Any]] = []
    for record in reversed(store.list("datasets")):
        source = _flat(record)
        if source.get("zone_id") != zone_id:
            continue
        if family and source.get("family") != family:
            continue
        for item in source.get("expected_pieces") or []:
            if isinstance(item, dict):
                dataset_pieces.append(_expected_piece_dict(item))
        if dataset_pieces:
            return dataset_pieces

    config_path = Path("config/inspection.json")
    if not config_path.exists():
        return []
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return []
    families = config.get("families") or {}
    family_items = [(family, families.get(family))] if family and family in families else list(families.items())
    for _, family_config in family_items:
        if not isinstance(family_config, dict):
            continue
        zones = family_config.get("zones") or {}
        zone_config = None
        for candidate_zone_id in _zone_config_candidates(zone_id):
            zone_config = zones.get(candidate_zone_id)
            if isinstance(zone_config, dict):
                break
        if not isinstance(zone_config, dict):
            continue
        return [_expected_piece_dict(item) for item in zone_config.get("expected") or [] if isinstance(item, dict)]
    return []


def object_public_url(object_uri: str) -> str:
    if object_uri.startswith("local://"):
        upload_id = object_uri.removeprefix("local://")
        return f"/v1/uploads/{upload_id}/file"
    if object_uri.startswith("file://"):
        return object_uri
    return object_uri


def _upsert_zone_reference(zone_id: str, reference: dict[str, Any], store: MetadataStore) -> None:
    family = str(reference["family"])
    zone_record_id = f"{family}:{zone_id}"
    existing = store.get("zones", zone_record_id) or store.get("zones", zone_id) or {}
    source = _flat(existing)
    payload = {
        **source,
        "id": zone_record_id,
        "family": family,
        "zone_id": zone_id,
        "reference_id": reference["reference_id"],
        "reference_image_uri": reference["image_uri"],
        "reference_image_url": reference.get("image_url"),
        "mask_uri": reference.get("mask_uri"),
        "mask_url": reference.get("mask_url"),
        "tolerance": reference.get("tolerance") or DEFAULT_TOLERANCE,
        "updated_at": utc_now(),
    }
    store.put("zones", zone_record_id, payload)


def _expected_piece_dict(item: dict[str, Any]) -> dict[str, Any]:
    class_name = str(item.get("class_name") or item.get("name") or item.get("id") or "piece")
    record = ExpectedPieceRecord(
        id=str(item.get("id") or class_name),
        class_name=class_name,
        name=str(item.get("name") or item.get("label") or item.get("id") or class_name),
        roi=item.get("roi") if isinstance(item.get("roi"), list) else None,
        required=bool(item.get("required", item.get("critical", True))),
        critical=bool(item.get("critical", item.get("required", True))),
    )
    return record.model_dump()


def _with_public_urls(record: dict[str, Any]) -> dict[str, Any]:
    next_record = dict(record)
    if next_record.get("image_uri"):
        next_record["image_url"] = object_public_url(str(next_record["image_uri"]))
    if next_record.get("mask_uri"):
        next_record["mask_url"] = object_public_url(str(next_record["mask_uri"]))
    if "tolerance" not in next_record or not isinstance(next_record["tolerance"], dict):
        next_record["tolerance"] = DEFAULT_TOLERANCE
    return next_record


def _reference_record_id(family: str, zone_id: str, reference_id: str) -> str:
    return f"ref_{_safe_token(family)}_{_safe_token(zone_id)}_{_safe_token(reference_id)}"


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower() or "default"


def _zone_config_candidates(zone_id: str) -> list[str]:
    candidates = [zone_id]
    match = re.fullmatch(r"zona_(\d+)_(front|left|right)", zone_id)
    if match and match.group(2) == "front":
        candidates.append(f"frontal_zona_{match.group(1)}")
    return candidates


def _flat(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    return data if isinstance(data, dict) else record


def _image_size(path: Path) -> tuple[int | None, int | None]:
    if cv2 is None:
        return None, None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return int(width), int(height)
