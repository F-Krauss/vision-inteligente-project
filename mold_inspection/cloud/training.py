from __future__ import annotations

from typing import Any

from .config import CloudSettings
from .schemas import TrainingJobCreateRequest, TrainingJobRecord
from .store import MetadataStore


def create_training_job(
    request: TrainingJobCreateRequest,
    settings: CloudSettings,
    store: MetadataStore,
) -> TrainingJobRecord:
    record = TrainingJobRecord(
        family=request.family,
        zone_id=request.zone_id,
        dataset_uri=request.dataset_uri,
        manifest_uri=request.manifest_uri,
        mask_uri=request.mask_uri,
        output_uri=request.output_uri,
        target=request.target,
        message="Trabajo registrado. Vertex AI no esta habilitado en este entorno.",
        request=request.model_dump(),
    )
    if settings.enable_vertex_training:
        vertex = _submit_vertex_custom_job(request, settings)
        record.status = vertex["status"]
        record.vertex_job_name = vertex.get("vertex_job_name")
        record.message = vertex["message"]
    store.put("training_jobs", record.id, record.model_dump())
    return record


def _submit_vertex_custom_job(request: TrainingJobCreateRequest, settings: CloudSettings) -> dict[str, Any]:
    if not settings.vertex_training_image:
        return {
            "status": "requires_configuration",
            "message": "Falta MOLD_VERTEX_TRAINING_IMAGE para enviar el CustomJob a Vertex AI.",
        }
    if not request.manifest_uri:
        return {
            "status": "requires_manifest",
            "message": "Falta manifest_uri para entrenar en Vertex AI.",
        }
    if not request.mask_uri:
        return {
            "status": "requires_mask",
            "message": "Falta mask_uri para entrenar esta zona en Vertex AI.",
        }
    try:
        from google.cloud import aiplatform
    except ImportError as exc:  # pragma: no cover
        return {
            "status": "requires_dependency",
            "message": f"google-cloud-aiplatform no esta instalado: {exc}",
        }

    aiplatform.init(
        project=settings.project_id,
        location=settings.region,
        staging_bucket=settings.vertex_staging_bucket,
    )
    staging_bucket = settings.vertex_staging_bucket
    if not staging_bucket:
        return {
            "status": "requires_configuration",
            "message": "Falta MOLD_VERTEX_STAGING_BUCKET o MOLD_ARTIFACT_BUCKET para guardar artefactos.",
        }
    output_uri = request.output_uri or f"{staging_bucket.rstrip('/')}/models/{request.family}/{request.zone_id}"
    args = [
        "--family",
        request.family,
        "--zone-id",
        request.zone_id,
        "--manifest-uri",
        request.manifest_uri,
        "--mask-uri",
        request.mask_uri,
        "--output-uri",
        output_uri,
        "--target",
        "cloud-gpu",
    ]
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
        display_name=f"mold-{request.family}-{request.zone_id}",
        worker_pool_specs=worker_pool_specs,
    )
    job.run(
        service_account=settings.vertex_service_account,
        sync=False,
    )
    return {
        "status": "submitted",
        "vertex_job_name": getattr(job, "resource_name", None),
        "message": "CustomJob enviado a Vertex AI.",
    }
