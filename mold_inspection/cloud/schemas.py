from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


InspectionStatus = Literal["correct", "review", "retake_photo"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class UploadPresignRequest(BaseModel):
    filename: str
    content_type: str = "image/jpeg"
    family: str | None = None
    zone_id: str | None = None
    purpose: Literal["inspection", "dataset", "segmenter"] = "inspection"


class UploadPresignResponse(BaseModel):
    upload_id: str
    object_uri: str
    upload_url: str
    method: str = "PUT"
    headers: dict[str, str] = Field(default_factory=dict)
    expires_at: str


class InspectionCreateRequest(BaseModel):
    family: str
    zone_id: str
    image_uri: str
    mold_id: str | None = None
    session_id: str | None = None
    operator_id: str | None = None
    capture_metadata: dict[str, Any] = Field(default_factory=dict)


class InspectionRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("insp"))
    created_at: str = Field(default_factory=utc_now)
    family: str
    zone_id: str
    image_uri: str
    mold_id: str | None = None
    session_id: str | None = None
    operator_id: str | None = None
    status: InspectionStatus
    message: str
    result: dict[str, Any] = Field(default_factory=dict)
    guidance: list[str] = Field(default_factory=list)
    evidence: dict[str, str] = Field(default_factory=dict)
    identified_mold: str | None = None
    identified_zone: str | None = None
    confidence: float | None = None
    mold_polygon: list[dict[str, float]] = Field(default_factory=list)
    missing_regions: list[list[dict[str, float]]] = Field(default_factory=list)
    overlay_image_uri: str | None = None


class TrainingJobCreateRequest(BaseModel):
    family: str
    zone_id: str
    dataset_uri: str
    manifest_uri: str | None = None
    mask_uri: str | None = None
    output_uri: str | None = None
    target: Literal["cloud-gpu"] = "cloud-gpu"
    notes: str | None = None


class TrainingJobRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("train"))
    created_at: str = Field(default_factory=utc_now)
    family: str
    zone_id: str
    dataset_uri: str
    manifest_uri: str | None = None
    mask_uri: str | None = None
    output_uri: str | None = None
    target: str = "cloud-gpu"
    status: str = "queued"
    vertex_job_name: str | None = None
    message: str
    request: dict[str, Any] = Field(default_factory=dict)


class MaskPoint(BaseModel):
    x: float
    y: float


class DatasetMaskPayload(BaseModel):
    type: Literal["auto", "polygon", "png_uri"] = "auto"
    points: list[MaskPoint] | None = None
    png_uri: str | None = None


class ExpectedPiecePayload(BaseModel):
    id: str
    name: str | None = None
    class_name: str
    required: bool = True
    region: list[MaskPoint] | None = None


class DatasetFromExamplesRequest(BaseModel):
    family: str
    zone_id: str
    name: str = "Dataset de referencia"
    ok_image_uris: list[str]
    fault_image_uris: list[str]
    mask: DatasetMaskPayload = Field(default_factory=DatasetMaskPayload)
    expected_pieces: list[ExpectedPiecePayload] = Field(default_factory=list)


class DatasetFromExamplesResponse(BaseModel):
    id: str
    created_at: str = Field(default_factory=utc_now)
    family: str
    zone_id: str
    name: str
    status: str = "ready_for_training"
    manifest_uri: str
    mask_uri: str
    dataset_uri: str
    ok_count: int
    fault_count: int
    piece_count: int = 0
    preview_image_uri: str | None = None


class CaptureGuidanceRequest(BaseModel):
    family: str
    zone_id: str
    image_uri: str


class CaptureGuidanceResponse(BaseModel):
    ok: bool
    auto_capture_ready: bool = False
    message: str
    guidance: list[str] = Field(default_factory=list)
    quality: dict[str, Any] = Field(default_factory=dict)
    alignment: dict[str, Any] = Field(default_factory=dict)


class SegmenterAnnotation(BaseModel):
    image_uri: str
    polygon: list[MaskPoint]
    split: Literal["train", "val", "test"] = "train"


class SegmenterDatasetCreateRequest(BaseModel):
    name: str = "Dataset segmentador de moldes"
    annotations: list[SegmenterAnnotation]


class SegmenterDatasetRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("segds"))
    created_at: str = Field(default_factory=utc_now)
    name: str
    status: str = "ready_for_training"
    dataset_uri: str
    data_yaml_uri: str
    image_count: int
    train_count: int
    val_count: int
    test_count: int


class SegmenterTrainingJobCreateRequest(BaseModel):
    dataset_id: str
    base_model: str = "yolov8n-seg.pt"
    epochs: int = 50
    image_size: int = 640
    output_uri: str | None = None


class SegmenterTrainingJobRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("segtrain"))
    created_at: str = Field(default_factory=utc_now)
    dataset_id: str
    status: str = "queued"
    message: str
    model_uri: str | None = None
    onnx_uri: str | None = None
    data_yaml_uri: str | None = None
    training_command: list[str] = Field(default_factory=list)
    request: dict[str, Any] = Field(default_factory=dict)


class RecipeCreateRequest(BaseModel):
    family: str
    zone_id: str
    name: str
    mold_id: str | None = None
    objective: Literal["presence_absence"] = "presence_absence"
    notes: str | None = None


class RecipeRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("recipe"))
    created_at: str = Field(default_factory=utc_now)
    family: str
    zone_id: str
    name: str
    mold_id: str | None = None
    objective: str = "presence_absence"
    status: str = "draft"
    dataset_id: str | None = None
    segmenter_job_id: str | None = None
    inspector_job_id: str | None = None
    best_model_candidate_id: str | None = None
    readiness: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class InspectorTrainingJobCreateRequest(BaseModel):
    family: str
    zone_id: str
    dataset_id: str | None = None
    dataset_uri: str | None = None
    manifest_uri: str | None = None
    mask_uri: str | None = None
    target: Literal["cloud-gpu"] = "cloud-gpu"
    notes: str | None = None


class InspectorTrainingJobRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("insptrain"))
    created_at: str = Field(default_factory=utc_now)
    family: str
    zone_id: str
    dataset_id: str | None = None
    dataset_uri: str | None = None
    manifest_uri: str | None = None
    mask_uri: str | None = None
    target: str = "cloud-gpu"
    status: str = "queued"
    message: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    best_model_candidate_id: str | None = None
    training_command: list[str] = Field(default_factory=list)
    request: dict[str, Any] = Field(default_factory=dict)


class ModelCandidatePromotionRequest(BaseModel):
    notes: str | None = None


class PublicDatasetImportRequest(BaseModel):
    dataset: Literal["mvtec_ad", "visa", "kolektor_sdd", "abo"]
    category: str | None = None
    local_root: str | None = None
    max_items: int = 50
    family: str | None = None
    zone_id: str | None = None


class PublicDatasetImportRecord(BaseModel):
    id: str = Field(default_factory=lambda: new_id("pubds"))
    created_at: str = Field(default_factory=utc_now)
    dataset: str
    category: str | None = None
    status: str
    source_url: str
    license: str
    license_url: str
    intended_use: str = "benchmark_only"
    family: str | None = None
    zone_id: str | None = None
    manifest_uri: str | None = None
    mask_uri: str | None = None
    ok_count: int = 0
    fault_count: int = 0
    mask_count: int = 0
    message: str
    warnings: list[str] = Field(default_factory=list)


class ResourceRecord(BaseModel):
    id: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
