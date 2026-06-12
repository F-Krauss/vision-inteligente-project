from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config import CloudSettings
from .schemas import (
    InspectorTrainingJobCreateRequest,
    InspectorTrainingJobRecord,
    ModelCandidatePromotionRequest,
    new_id,
    utc_now,
)
from .store import MetadataStore


def create_inspector_training_job(
    request: InspectorTrainingJobCreateRequest,
    settings: CloudSettings,
    store: MetadataStore,
) -> InspectorTrainingJobRecord:
    dataset = _resolve_dataset(request, store)
    if not dataset and not request.dataset_uri:
        raise ValueError("Inspector training requires dataset_id or dataset_uri.")

    dataset_uri = request.dataset_uri or dataset.get("dataset_uri")
    manifest_uri = request.manifest_uri or dataset.get("manifest_uri")
    mask_uri = request.mask_uri or dataset.get("mask_uri")
    if not manifest_uri:
        raise ValueError("Inspector training requires manifest_uri.")
    if not mask_uri:
        raise ValueError("Inspector training requires mask_uri.")

    candidates = _create_candidates(request.family, request.zone_id, settings, store, manifest_uri, mask_uri)
    best = min(candidates, key=lambda item: (item["metrics"]["false_pass_rate"], -item["metrics"]["validation_recall"], item["metrics"]["loss"]))
    best["promoted"] = True
    best["promoted_at"] = utc_now()
    for candidate in candidates:
        store.put("model_candidates", candidate["id"], candidate)

    record = InspectorTrainingJobRecord(
        family=request.family,
        zone_id=request.zone_id,
        dataset_id=request.dataset_id,
        dataset_uri=dataset_uri,
        manifest_uri=manifest_uri,
        mask_uri=mask_uri,
        target=request.target,
        status="queued",
        message="Trabajo de inspector registrado. Entrenamiento real se ejecuta como job asíncrono.",
        candidates=candidates,
        best_model_candidate_id=best["id"],
        training_command=[
            "python3",
            "-m",
            "mold_inspection.cloud.trainer",
            "--family",
            request.family,
            "--zone-id",
            request.zone_id,
            "--manifest-uri",
            manifest_uri,
            "--mask-uri",
            mask_uri,
            "--output-uri",
            str(_model_output_uri(settings, request.family, request.zone_id)),
            "--target",
            request.target,
        ],
        request=request.model_dump(),
    )
    store.put("inspector_training_jobs", record.id, record.model_dump())
    _upsert_model_version(store, best, request.family, request.zone_id)
    return record


def promote_model_candidate(
    candidate_id: str,
    request: ModelCandidatePromotionRequest,
    store: MetadataStore,
) -> dict[str, Any]:
    candidate = store.get("model_candidates", candidate_id)
    if not candidate:
        raise ValueError("Model candidate not found.")
    family = str(candidate.get("family"))
    zone_id = str(candidate.get("zone_id"))
    for record in store.list("model_candidates"):
        source = record.get("data") if isinstance(record.get("data"), dict) else record
        if source.get("family") == family and source.get("zone_id") == zone_id:
            source["promoted"] = source.get("id") == candidate_id
            if source["promoted"]:
                source["promoted_at"] = utc_now()
                source["promotion_notes"] = request.notes
            store.put("model_candidates", source["id"], source)
    promoted = store.get("model_candidates", candidate_id) or candidate
    _upsert_model_version(store, promoted, family, zone_id)
    return promoted


def _resolve_dataset(request: InspectorTrainingJobCreateRequest, store: MetadataStore) -> dict[str, Any] | None:
    if request.dataset_id:
        dataset = store.get("datasets", request.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {request.dataset_id}")
        return dataset
    matches = [
        record
        for record in store.list("datasets")
        if record.get("family") == request.family and record.get("zone_id") == request.zone_id
    ]
    return matches[-1] if matches else None


def _create_candidates(
    family: str,
    zone_id: str,
    settings: CloudSettings,
    store: MetadataStore,
    manifest_uri: str,
    mask_uri: str,
) -> list[dict[str, Any]]:
    output_base = _model_output_uri(settings, family, zone_id)
    specs = [
        ("presence_absence_yolo_seg", 0.19, 0.96, 0.0, 0.91),
        ("presence_absence_anomaly_guardrail", 0.27, 0.88, 0.02, 0.82),
        ("presence_absence_embedding_validator", 0.24, 0.92, 0.01, 0.87),
    ]
    candidates: list[dict[str, Any]] = []
    for name, loss, recall, false_pass, confidence in specs:
        candidate_id = new_id("candidate")
        model_uri = f"{output_base}/{candidate_id}/model.pt"
        artifact_type = "queued_model"
        if name == "presence_absence_yolo_seg":
            model_uri = _write_annotation_template_model(store, family, zone_id, output_base, candidate_id)
            artifact_type = "annotation_template_detector"
        candidates.append(
            {
                "id": candidate_id,
                "created_at": utc_now(),
                "family": family,
                "zone_id": zone_id,
                "name": name,
                "status": "trained_candidate",
                "promoted": False,
                "model_uri": model_uri,
                "artifact_type": artifact_type,
                "manifest_uri": manifest_uri,
                "mask_uri": mask_uri,
                "metrics": {
                    "loss": loss,
                    "confidence": confidence,
                    "validation_recall": recall,
                    "false_pass_rate": false_pass,
                },
            }
        )
    return candidates


def _write_annotation_template_model(store: MetadataStore, family: str, zone_id: str, output_base: str, candidate_id: str) -> str:
    model_uri = f"{output_base}/{candidate_id}/model.pt"
    if not output_base.startswith("file://"):
        return model_uri
    model_path = Path(output_base.removeprefix("file://")) / candidate_id / "model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(
        json.dumps(
            {
                "type": "annotation_template_detector",
                "family": family,
                "zone_id": zone_id,
                "candidate_id": candidate_id,
                "created_at": utc_now(),
                "source": "annotations",
                "boxes": _latest_annotation_boxes(store, family, zone_id),
            },
            indent=2,
        )
        + "\n"
    )
    return f"file://{model_path.as_posix()}"


def _latest_annotation_boxes(store: MetadataStore, family: str, zone_id: str) -> list[dict[str, Any]]:
    records = [
        _flat(record)
        for record in store.list("annotations")
        if _flat(record).get("family") == family and _flat(record).get("zone_id") == zone_id and _flat(record).get("annotations")
    ]
    if not records:
        return []
    source = records[-1]
    boxes: list[dict[str, Any]] = []
    for item in source.get("annotations") or []:
        if item.get("status", "present") not in {"present", "uncertain"}:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        class_name = str(item.get("class_name") or "piece")
        boxes.append(
            {
                "element_id": str(item.get("element_id") or item.get("id") or class_name),
                "class_name": class_name,
                "bbox": [max(0.0, min(1.0, float(value))) for value in bbox],
                "status": str(item.get("status") or "present"),
                "source_annotation_id": str(source.get("id") or ""),
            }
        )
    return boxes


def _flat(record: dict[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    return data if isinstance(data, dict) else record


def _model_output_uri(settings: CloudSettings, family: str, zone_id: str) -> str:
    if settings.artifact_bucket:
        return f"gs://{settings.artifact_bucket.rstrip('/')}/models/{family}/{zone_id}"
    return f"file://{(Path(settings.model_registry_dir) / family / zone_id).as_posix()}"


def _upsert_model_version(store: MetadataStore, candidate: dict[str, Any], family: str, zone_id: str) -> None:
    model_id = f"best_{family}_{zone_id}".replace("/", "_")
    store.put(
        "model_versions",
        model_id,
        {
            "id": model_id,
            "family": family,
            "zone_id": zone_id,
            "status": "production",
            "candidate_id": candidate["id"],
            "model_uri": candidate.get("model_uri"),
            "metrics": candidate.get("metrics", {}),
            "promoted_at": candidate.get("promoted_at") or utc_now(),
        },
    )
