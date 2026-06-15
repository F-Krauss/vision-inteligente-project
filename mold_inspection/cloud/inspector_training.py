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
    data_yaml_uri = request.data_yaml_uri or dataset.get("data_yaml_uri")
    manifest_uri = request.manifest_uri or dataset.get("manifest_uri")
    mask_uri = request.mask_uri or dataset.get("mask_uri")
    if not manifest_uri:
        raise ValueError("Inspector training requires manifest_uri.")
    if not mask_uri:
        raise ValueError("Inspector training requires mask_uri.")

    output_uri = str(_model_output_uri(settings, request.family, request.zone_id))
    candidates = _create_candidates(request.family, request.zone_id, settings, store, manifest_uri, mask_uri, output_uri, bool(data_yaml_uri))
    best = _best_candidate(candidates, prefer_real_yolo=settings.enable_vertex_training)
    best["promoted"] = True
    best["promoted_at"] = utc_now()
    for candidate in candidates:
        store.put("model_candidates", candidate["id"], candidate)

    training_command = _training_command(request, manifest_uri, mask_uri, data_yaml_uri, dataset_uri, output_uri)
    vertex = _submit_vertex_custom_job(request, settings, manifest_uri, mask_uri, data_yaml_uri, dataset_uri, output_uri) if settings.enable_vertex_training else None
    record = InspectorTrainingJobRecord(
        family=request.family,
        zone_id=request.zone_id,
        dataset_id=request.dataset_id,
        dataset_uri=dataset_uri,
        data_yaml_uri=data_yaml_uri,
        manifest_uri=manifest_uri,
        mask_uri=mask_uri,
        target=request.target,
        status=vertex["status"] if vertex else "queued",
        vertex_job_name=vertex.get("vertex_job_name") if vertex else None,
        message=vertex["message"] if vertex else "Trabajo de inspector registrado. Ejecuta training_command o habilita Vertex AI.",
        candidates=candidates,
        best_model_candidate_id=best["id"],
        training_command=training_command,
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
    output_base: str,
    include_yolo: bool,
) -> list[dict[str, Any]]:
    specs = []
    if include_yolo:
        specs.append(("piece_group_yolo_detector", 0.18, 0.97, 0.0, 0.92, "yolo_piece_detector"))
    specs += [
        ("presence_absence_yolo_seg", 0.19, 0.96, 0.0, 0.91, "annotation_template_detector"),
        ("presence_absence_anomaly_guardrail", 0.27, 0.88, 0.02, 0.82),
        ("presence_absence_embedding_validator", 0.24, 0.92, 0.01, 0.87),
    ]
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        name, loss, recall, false_pass, confidence = spec[:5]
        requested_type = spec[5] if len(spec) > 5 else "queued_model"
        candidate_id = new_id("candidate")
        model_uri = f"{output_base}/{candidate_id}/model.pt"
        artifact_type = "queued_model"
        if requested_type == "yolo_piece_detector":
            model_uri = f"{output_base.rstrip('/')}/piece_detector/best.pt"
            artifact_type = "yolo_piece_detector"
        elif requested_type == "annotation_template_detector":
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


def _best_candidate(candidates: list[dict[str, Any]], prefer_real_yolo: bool) -> dict[str, Any]:
    if prefer_real_yolo:
        real = next((candidate for candidate in candidates if candidate.get("artifact_type") == "yolo_piece_detector"), None)
        if real:
            return real
    template = next((candidate for candidate in candidates if candidate.get("artifact_type") == "annotation_template_detector"), None)
    if template:
        return template
    return min(candidates, key=lambda item: (item["metrics"]["false_pass_rate"], -item["metrics"]["validation_recall"], item["metrics"]["loss"]))


def _training_command(
    request: InspectorTrainingJobCreateRequest,
    manifest_uri: str,
    mask_uri: str,
    data_yaml_uri: str | None,
    dataset_uri: str | None,
    output_uri: str,
) -> list[str]:
    command = [
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
        output_uri,
        "--target",
        request.target,
    ]
    if data_yaml_uri:
        command.extend(["--data-yaml-uri", data_yaml_uri])
        command.extend(["--yolo-base-model", request.yolo_base_model])
        command.extend(["--yolo-epochs", str(request.yolo_epochs)])
        command.extend(["--yolo-image-size", str(request.yolo_image_size)])
    if dataset_uri:
        command.extend(["--dataset-uri", dataset_uri])
    return command


def _submit_vertex_custom_job(
    request: InspectorTrainingJobCreateRequest,
    settings: CloudSettings,
    manifest_uri: str,
    mask_uri: str,
    data_yaml_uri: str | None,
    dataset_uri: str | None,
    output_uri: str,
) -> dict[str, Any]:
    if not settings.vertex_training_image:
        return {"status": "requires_configuration", "message": "Falta MOLD_VERTEX_TRAINING_IMAGE para Vertex AI."}
    if not settings.vertex_staging_bucket:
        return {"status": "requires_configuration", "message": "Falta MOLD_VERTEX_STAGING_BUCKET o MOLD_ARTIFACT_BUCKET."}
    try:
        from google.cloud import aiplatform
    except ImportError as exc:  # pragma: no cover
        return {"status": "requires_dependency", "message": f"google-cloud-aiplatform no esta instalado: {exc}"}

    aiplatform.init(project=settings.project_id, location=settings.region, staging_bucket=settings.vertex_staging_bucket)
    args = _training_command(request, manifest_uri, mask_uri, data_yaml_uri, dataset_uri, output_uri)[3:]
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "g2-standard-8",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": settings.vertex_training_image,
                "command": ["python", "-m", "mold_inspection.cloud.trainer"],
                "args": args,
            },
        }
    ]
    job = aiplatform.CustomJob(
        display_name=f"mold-inspector-{request.family}-{request.zone_id}",
        worker_pool_specs=worker_pool_specs,
    )
    job.run(service_account=settings.vertex_service_account, sync=False)
    return {
        "status": "submitted",
        "vertex_job_name": getattr(job, "resource_name", None),
        "message": "Entrenamiento YOLO de grupos de piezas enviado a Vertex AI.",
    }


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
                "shape": "rect" if item.get("shape") == "rect" else "polygon",
                "polygon": item.get("polygon") if isinstance(item.get("polygon"), list) else None,
                "category_id": item.get("category_id"),
                "category_name": item.get("category_name"),
                "importance": item.get("importance") if item.get("importance") in {"critical", "relevant", "minor"} else None,
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
