from __future__ import annotations

from pathlib import Path

from .config import CloudSettings


def materialize_model_uri(model_uri: str, settings: CloudSettings, filename: str | None = None) -> Path | None:
    if not model_uri:
        return None
    if model_uri.startswith("file://"):
        path = Path(model_uri.removeprefix("file://"))
        return path if path.exists() else None
    if not model_uri.startswith("gs://"):
        path = Path(model_uri)
        return path if path.exists() else None
    try:
        from google.cloud import storage
    except ImportError:  # pragma: no cover
        return None
    bucket_name, blob_name = model_uri.removeprefix("gs://").split("/", 1)
    suffix = Path(blob_name).suffix or Path(filename or "model.pt").suffix or ".pt"
    safe_name = filename or Path(blob_name).name or f"model{suffix}"
    destination = settings.local_state_dir / "model_cache" / bucket_name / blob_name
    if destination.name != safe_name and not destination.suffix:
        destination = destination / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    storage.Client(project=settings.project_id).bucket(bucket_name).blob(blob_name).download_to_filename(destination)
    return destination if destination.exists() else None
