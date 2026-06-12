from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .annotations import create_auto_annotation_draft, create_dataset_from_annotations, list_annotations, save_annotation
from .config import CloudSettings, load_settings
from .datasets import create_dataset_from_examples
from .guidance import create_capture_guidance
from .inspector_training import create_inspector_training_job, promote_model_candidate
from .pipeline import CloudInspectionPipeline
from .public_datasets import import_public_dataset
from .references import create_zone_reference, expected_pieces_for_zone, get_zone_reference
from .recipes import create_recipe, list_recipes
from .segmenter import create_segmenter_dataset, create_segmenter_training_job
from .section_plans import get_mold_section_plan, list_mold_section_plans, upsert_mold_section_plan
from .schemas import (
    AlignQualityRequest,
    AlignQualityResponse,
    AnnotationCreateRequest,
    AutoAnnotationDraftRequest,
    CaptureGuidanceRequest,
    DatasetFromExamplesRequest,
    DatasetFromAnnotationsRequest,
    InspectorTrainingJobCreateRequest,
    InspectionCreateRequest,
    ModelCandidatePromotionRequest,
    MoldSectionPlanUpsertRequest,
    MoldValidationSectionUpdateRequest,
    MoldValidationSessionCreateRequest,
    PublicDatasetImportRequest,
    ReferenceCreateRequest,
    RecipeCreateRequest,
    SegmenterDatasetCreateRequest,
    SegmenterTrainingJobCreateRequest,
    TrainingJobCreateRequest,
    UploadPresignRequest,
)
from .storage import create_object_storage
from .store import create_store, upsert_resource
from .training import create_training_job
from .validation_sessions import (
    create_mold_validation_session,
    get_mold_validation_session,
    list_mold_validation_sessions,
    update_mold_validation_section,
)


def create_app(settings: CloudSettings | None = None) -> FastAPI:
    settings = settings or load_settings()
    settings.local_state_dir.mkdir(parents=True, exist_ok=True)
    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    store = create_store(settings)
    objects = create_object_storage(settings, store)
    pipeline = CloudInspectionPipeline(settings, store, objects)

    app = FastAPI(
        title="Mold Vision Cloud API",
        version="0.1.0",
        description="Cloud-first visual inspection API for industrial molds.",
    )
    cors_origins = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": settings.service_name,
            "project_id": settings.project_id,
            "region": settings.region,
            "metadata_backend": settings.metadata_backend,
            "storage": "gcs" if settings.storage_bucket and settings.metadata_backend == "firestore" else "local",
        }

    @app.post("/v1/uploads/presign")
    def presign_upload(request: UploadPresignRequest) -> dict[str, Any]:
        return objects.create_upload(
            filename=request.filename,
            content_type=request.content_type,
            family=request.family,
            zone_id=request.zone_id,
            purpose=request.purpose,
        ).model_dump()

    @app.put("/v1/uploads/{upload_id}")
    async def upload_bytes(upload_id: str, request: Request) -> dict[str, Any]:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Upload body is empty")
        object_uri = objects.write_upload(upload_id, body, request.headers.get("content-type"))
        return {"upload_id": upload_id, "object_uri": object_uri}

    @app.get("/v1/uploads/{upload_id}/file")
    def get_upload_file(upload_id: str) -> FileResponse:
        record = store.get("uploads", upload_id)
        if not record:
            raise HTTPException(status_code=404, detail="Upload not found")
        path = Path(str(record.get("path", "")))
        if not path.exists():
            raise HTTPException(status_code=404, detail="Upload file not found")
        return FileResponse(path, media_type=str(record.get("content_type") or "application/octet-stream"))

    @app.post("/v1/inspections")
    def create_inspection(request: InspectionCreateRequest) -> dict[str, Any]:
        try:
            return pipeline.inspect(request).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/capture-guidance")
    def capture_guidance(request: CaptureGuidanceRequest) -> dict[str, Any]:
        try:
            return create_capture_guidance(request, objects, store, settings.model_registry_dir).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/uploads/align-quality")
    def align_quality(request: AlignQualityRequest) -> dict[str, Any]:
        try:
            guidance = create_capture_guidance(
                CaptureGuidanceRequest(
                    family=request.family,
                    zone_id=request.zone_id,
                    image_uri=request.image_uri,
                    reference_id=request.reference_id,
                ),
                objects,
                store,
                settings.model_registry_dir,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AlignQualityResponse(
            status="correct" if guidance.ok else "retake_photo",
            ok=guidance.ok,
            auto_capture_ready=guidance.auto_capture_ready,
            message=guidance.message if guidance.ok else "Retoma la foto: " + guidance.message,
            guidance=guidance.guidance,
            quality=guidance.quality,
            alignment=guidance.alignment,
        ).model_dump()

    @app.get("/v1/zones/{zone_id}/reference")
    def get_reference(zone_id: str, family: str | None = None, reference_id: str | None = None) -> dict[str, Any]:
        record = get_zone_reference(zone_id, store, family=family, reference_id=reference_id)
        if not record:
            raise HTTPException(status_code=404, detail="Reference not found")
        return record

    @app.post("/v1/zones/{zone_id}/reference")
    def post_reference(zone_id: str, request: ReferenceCreateRequest) -> dict[str, Any]:
        try:
            return create_zone_reference(zone_id, request, objects, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/zones/{zone_id}/expected")
    def get_expected(zone_id: str, family: str | None = None) -> list[dict[str, Any]]:
        return expected_pieces_for_zone(zone_id, store, family=family)

    @app.get("/v1/inspections")
    def list_inspections() -> list[dict[str, Any]]:
        return store.list("inspections")

    @app.get("/v1/inspections/{inspection_id}")
    def get_inspection(inspection_id: str) -> dict[str, Any]:
        record = store.get("inspections", inspection_id)
        if not record:
            raise HTTPException(status_code=404, detail="Inspection not found")
        return record

    @app.get("/v1/mold-section-plans")
    def get_section_plans(family: str | None = None) -> list[dict[str, Any]]:
        return list_mold_section_plans(store, family=family)

    @app.get("/v1/mold-section-plans/{mold_key}")
    def get_section_plan(mold_key: str, family: str | None = None) -> dict[str, Any]:
        record = get_mold_section_plan(store, mold_key, family=family)
        if not record:
            raise HTTPException(status_code=404, detail="Section plan not found")
        return record

    @app.post("/v1/mold-section-plans/{mold_key}")
    def post_section_plan(mold_key: str, request: MoldSectionPlanUpsertRequest) -> dict[str, Any]:
        try:
            return upsert_mold_section_plan(store, mold_key, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/v1/mold-section-plans/{mold_key}")
    def put_section_plan(mold_key: str, request: MoldSectionPlanUpsertRequest) -> dict[str, Any]:
        try:
            return upsert_mold_section_plan(store, mold_key, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/mold-validation-sessions")
    def get_validation_sessions(
        family: str | None = None,
        mold_key: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        return list_mold_validation_sessions(store, family=family, mold_key=mold_key, status=status)

    @app.post("/v1/mold-validation-sessions")
    def post_validation_session(request: MoldValidationSessionCreateRequest) -> dict[str, Any]:
        try:
            return create_mold_validation_session(store, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/mold-validation-sessions/{session_id}")
    def get_validation_session(session_id: str) -> dict[str, Any]:
        record = get_mold_validation_session(store, session_id)
        if not record:
            raise HTTPException(status_code=404, detail="Validation session not found")
        return record

    @app.post("/v1/mold-validation-sessions/{session_id}/sections/{section_id}")
    def post_validation_section(session_id: str, section_id: str, request: MoldValidationSectionUpdateRequest) -> dict[str, Any]:
        try:
            data = request.model_copy(update={"section_id": request.section_id or section_id})
            return update_mold_validation_section(store, session_id, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/training-jobs")
    def create_training(request: TrainingJobCreateRequest) -> dict[str, Any]:
        return create_training_job(request, settings, store).model_dump()

    @app.get("/v1/training-jobs")
    def list_training() -> list[dict[str, Any]]:
        return store.list("training_jobs")

    @app.get("/v1/training-jobs/{training_job_id}")
    def get_training(training_job_id: str) -> dict[str, Any]:
        record = store.get("training_jobs", training_job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Training job not found")
        return record

    @app.post("/v1/datasets/from-examples")
    def create_dataset(request: DatasetFromExamplesRequest) -> dict[str, Any]:
        try:
            return create_dataset_from_examples(request, objects, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/recipes")
    def create_ai_recipe(request: RecipeCreateRequest) -> dict[str, Any]:
        return create_recipe(request, store).model_dump()

    @app.get("/v1/recipes")
    def get_ai_recipes() -> list[dict[str, Any]]:
        return list_recipes(store)

    @app.post("/v1/segmenter-datasets/from-annotations")
    def create_mold_segmenter_dataset(request: dict[str, Any]) -> dict[str, Any]:
        try:
            if "annotations" in request:
                return create_segmenter_dataset(SegmenterDatasetCreateRequest(**request), settings, objects, store).model_dump()
            return create_dataset_from_annotations(DatasetFromAnnotationsRequest(**request), settings, objects, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/annotations")
    def create_annotation(request: AnnotationCreateRequest) -> dict[str, Any]:
        try:
            return save_annotation(request, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/annotations/auto-draft")
    def create_annotation_draft(request: AutoAnnotationDraftRequest) -> dict[str, Any]:
        try:
            return create_auto_annotation_draft(request, objects, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/annotations")
    def get_annotations(
        family: str | None = None,
        zone_id: str | None = None,
        image_id: str | None = None,
        image_uri: str | None = None,
    ) -> list[dict[str, Any]]:
        return list_annotations(store, family=family, zone_id=zone_id, image_id=image_id, image_uri=image_uri)

    @app.post("/v1/segmenter-training-jobs")
    def create_mold_segmenter_training(request: SegmenterTrainingJobCreateRequest) -> dict[str, Any]:
        try:
            return create_segmenter_training_job(request, settings, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/inspector-training-jobs")
    def create_piece_inspector_training(request: InspectorTrainingJobCreateRequest) -> dict[str, Any]:
        try:
            return create_inspector_training_job(request, settings, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/model-candidates/{candidate_id}/promote")
    def promote_candidate(candidate_id: str, request: ModelCandidatePromotionRequest | None = None) -> dict[str, Any]:
        try:
            return promote_model_candidate(candidate_id, request or ModelCandidatePromotionRequest(), store)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/public-datasets/import")
    def create_public_dataset_import(request: PublicDatasetImportRequest) -> dict[str, Any]:
        try:
            return import_public_dataset(request, objects, store).model_dump()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    for collection in [
        "annotation_datasets",
        "families",
        "molds",
        "zones",
        "references",
        "datasets",
        "segmenter_datasets",
        "segmenter_training_jobs",
        "inspector_training_jobs",
        "model_candidates",
        "public_dataset_imports",
        "model_versions",
    ]:
        _add_resource_routes(app, store, collection)

    settings.evidence_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/evidence", StaticFiles(directory=settings.evidence_dir), name="evidence")

    web_dist = Path(os.getenv("MOLD_WEB_DIST", "web/dist"))
    if web_dist.exists():
        assets = web_dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_web(full_path: str) -> FileResponse:
            target = web_dist / full_path
            if target.is_file():
                return FileResponse(target)
            return FileResponse(web_dist / "index.html")

    return app


def _cors_origins() -> list[str]:
    configured = os.getenv("MOLD_CORS_ORIGINS")
    if configured:
        separator = "|" if "|" in configured else ","
        return [origin.strip() for origin in configured.split(separator) if origin.strip()]
    return [
        "https://t-efficiency.com",
        "https://www.t-efficiency.com",
        "https://mold-vision-api-r52omw5uhq-uc.a.run.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]


def _add_resource_routes(app: FastAPI, store, collection: str) -> None:
    @app.get(f"/v1/{collection}", name=f"list_{collection}")
    def list_resources() -> list[dict[str, Any]]:
        return store.list(collection)

    @app.post(f"/v1/{collection}", name=f"upsert_{collection}")
    def upsert(data: dict[str, Any]) -> dict[str, Any]:
        try:
            return upsert_resource(store, collection, data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(f"/v1/{collection}/{{record_id}}", name=f"get_{collection}")
    def get_resource(record_id: str) -> dict[str, Any]:
        record = store.get(collection, record_id)
        if not record:
            raise HTTPException(status_code=404, detail="Resource not found")
        return record


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "mold_inspection.cloud.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("MOLD_RELOAD", "0") == "1",
    )
