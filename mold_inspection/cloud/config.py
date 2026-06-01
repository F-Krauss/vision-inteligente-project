from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class CloudSettings:
    project_id: str = "mia-prod"
    region: str = "us-central1"
    service_name: str = "mold-vision-api"
    storage_bucket: str | None = None
    artifact_bucket: str | None = None
    metadata_backend: str = "local"
    local_state_dir: Path = Path("data/cloud_state")
    model_registry_dir: Path = Path("data/model_registry")
    segmenter_model_path: Path = Path("data/segmenter/best.pt")
    anomaly_dir: Path = Path("data/anomaly")
    evidence_dir: Path = Path("reports/cloud_evidence")
    vertex_training_image: str | None = None
    vertex_staging_bucket: str | None = None
    vertex_service_account: str | None = None
    enable_vertex_training: bool = False
    public_base_url: str | None = None


def load_settings() -> CloudSettings:
    project_id = (
        os.getenv("MOLD_GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or "mia-prod"
    )
    local_state_dir = Path(os.getenv("MOLD_LOCAL_STATE_DIR", "data/cloud_state"))
    return CloudSettings(
        project_id=project_id,
        region=os.getenv("MOLD_GCP_REGION", "us-central1"),
        service_name=os.getenv("MOLD_SERVICE_NAME", "mold-vision-api"),
        storage_bucket=os.getenv("MOLD_UPLOAD_BUCKET"),
        artifact_bucket=os.getenv("MOLD_ARTIFACT_BUCKET"),
        metadata_backend=os.getenv("MOLD_METADATA_BACKEND", "local"),
        local_state_dir=local_state_dir,
        model_registry_dir=Path(os.getenv("MOLD_MODEL_REGISTRY_DIR", "data/model_registry")),
        segmenter_model_path=Path(os.getenv("MOLD_SEGMENTER_MODEL", "data/segmenter/best.pt")),
        anomaly_dir=Path(os.getenv("MOLD_ANOMALY_DIR", "data/anomaly")),
        evidence_dir=Path(os.getenv("MOLD_EVIDENCE_DIR", "reports/cloud_evidence")),
        vertex_training_image=os.getenv("MOLD_VERTEX_TRAINING_IMAGE"),
        vertex_staging_bucket=os.getenv("MOLD_VERTEX_STAGING_BUCKET") or os.getenv("MOLD_ARTIFACT_BUCKET"),
        vertex_service_account=os.getenv("MOLD_VERTEX_SERVICE_ACCOUNT"),
        enable_vertex_training=os.getenv("MOLD_ENABLE_VERTEX_TRAINING", "0") in {"1", "true", "TRUE", "yes"},
        public_base_url=os.getenv("MOLD_PUBLIC_BASE_URL"),
    )
