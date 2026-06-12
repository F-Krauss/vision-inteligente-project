from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import re
import shutil
import subprocess

from .config import CloudSettings
from .schemas import UploadPresignResponse
from .store import MetadataStore


class ObjectStorage:
    def create_upload(
        self,
        filename: str,
        content_type: str,
        family: str | None,
        zone_id: str | None,
        purpose: str,
    ) -> UploadPresignResponse:
        raise NotImplementedError

    def write_upload(self, upload_id: str, body: bytes, content_type: str | None = None) -> str:
        raise NotImplementedError

    def materialize(self, object_uri: str) -> Path:
        raise NotImplementedError


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: str | Path, store: MetadataStore):
        self.root = Path(root)
        self.store = store
        self.uploads_dir = self.root / "objects"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def create_upload(
        self,
        filename: str,
        content_type: str,
        family: str | None,
        zone_id: str | None,
        purpose: str,
    ) -> UploadPresignResponse:
        upload_id = f"upl_{uuid4().hex[:16]}"
        safe_name = _safe_filename(filename)
        key = Path(purpose) / (family or "unknown_family") / (zone_id or "unknown_zone") / upload_id / safe_name
        object_path = self.uploads_dir / key
        object_uri = f"local://{upload_id}"
        expires_at = _expires_at()
        self.store.put(
            "uploads",
            upload_id,
            {
                "id": upload_id,
                "object_uri": object_uri,
                "path": str(object_path),
                "filename": filename,
                "content_type": content_type,
                "purpose": purpose,
                "family": family,
                "zone_id": zone_id,
                "expires_at": expires_at,
            },
        )
        return UploadPresignResponse(
            upload_id=upload_id,
            object_uri=object_uri,
            upload_url=f"/v1/uploads/{upload_id}",
            headers={"Content-Type": content_type},
            expires_at=expires_at,
        )

    def write_upload(self, upload_id: str, body: bytes, content_type: str | None = None) -> str:
        record = self.store.get("uploads", upload_id)
        if not record:
            raise ValueError(f"Unknown upload id: {upload_id}")
        path = Path(record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        updated = {"uploaded": True, "size_bytes": len(body), "path": str(path)}
        if content_type:
            updated["content_type"] = content_type
        self.store.put("uploads", upload_id, {**record, **updated})
        return str(record["object_uri"])

    def materialize(self, object_uri: str) -> Path:
        if object_uri.startswith("local://"):
            upload_id = object_uri.removeprefix("local://")
            record = self.store.get("uploads", upload_id)
            if not record:
                raise ValueError(f"Unknown local object: {object_uri}")
            path = Path(record["path"])
            if not path.exists():
                raise ValueError(f"Upload has not been written yet: {object_uri}")
            return _ensure_cv_readable(path)
        path = Path(object_uri.removeprefix("file://"))
        if path.exists():
            return _ensure_cv_readable(path)
        raise ValueError(f"Unsupported or missing object URI: {object_uri}")


class GcsObjectStorage(ObjectStorage):
    def __init__(self, settings: CloudSettings, store: MetadataStore):
        if not settings.storage_bucket:
            raise ValueError("MOLD_UPLOAD_BUCKET is required for GCS storage")
        try:
            from google.cloud import storage
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install cloud dependencies to use Cloud Storage") from exc
        self.settings = settings
        self.store = store
        self.client = storage.Client(project=settings.project_id)
        self.bucket = self.client.bucket(settings.storage_bucket)
        self.cache_dir = settings.local_state_dir / "gcs_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def create_upload(
        self,
        filename: str,
        content_type: str,
        family: str | None,
        zone_id: str | None,
        purpose: str,
    ) -> UploadPresignResponse:
        upload_id = f"upl_{uuid4().hex[:16]}"
        safe_name = _safe_filename(filename)
        blob_name = f"{purpose}/{family or 'unknown_family'}/{zone_id or 'unknown_zone'}/{upload_id}/{safe_name}"
        blob = self.bucket.blob(blob_name)
        expires = datetime.now(timezone.utc) + timedelta(minutes=15)
        try:
            upload_url = blob.generate_signed_url(
                version="v4",
                expiration=expires,
                method="PUT",
                content_type=content_type,
            )
        except Exception:
            upload_url = f"/v1/uploads/{upload_id}"
        object_uri = f"gs://{self.bucket.name}/{blob_name}"
        self.store.put(
            "uploads",
            upload_id,
            {
                "id": upload_id,
                "object_uri": object_uri,
                "blob_name": blob_name,
                "filename": filename,
                "content_type": content_type,
                "purpose": purpose,
                "family": family,
                "zone_id": zone_id,
                "expires_at": expires.isoformat(),
            },
        )
        return UploadPresignResponse(
            upload_id=upload_id,
            object_uri=object_uri,
            upload_url=upload_url,
            headers={"Content-Type": content_type},
            expires_at=expires.isoformat(),
        )

    def write_upload(self, upload_id: str, body: bytes, content_type: str | None = None) -> str:
        record = self.store.get("uploads", upload_id)
        if not record:
            raise ValueError(f"Unknown upload id: {upload_id}")
        blob = self.bucket.blob(record["blob_name"])
        blob.upload_from_string(body, content_type=content_type or record.get("content_type"))
        self.store.put("uploads", upload_id, {**record, "uploaded": True, "size_bytes": len(body)})
        return str(record["object_uri"])

    def materialize(self, object_uri: str) -> Path:
        if not object_uri.startswith("gs://"):
            path = Path(object_uri.removeprefix("file://"))
            if path.exists():
                return path
            raise ValueError(f"Unsupported object URI: {object_uri}")
        bucket_name, blob_name = object_uri.removeprefix("gs://").split("/", 1)
        if bucket_name != self.bucket.name:
            raise ValueError(f"Unexpected bucket in URI: {object_uri}")
        destination = self.cache_dir / blob_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(blob_name).download_to_filename(destination)
        return _ensure_cv_readable(destination)


def create_object_storage(settings: CloudSettings, store: MetadataStore) -> ObjectStorage:
    if settings.storage_bucket and settings.metadata_backend == "firestore":
        return GcsObjectStorage(settings, store)
    return LocalObjectStorage(settings.local_state_dir, store)


def copy_to_uri(source: Path, destination_uri: str) -> None:
    if destination_uri.startswith("file://"):
        destination = Path(destination_uri.removeprefix("file://"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    raise ValueError(f"Unsupported destination URI: {destination_uri}")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name or "image.jpg"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _ensure_cv_readable(path: Path) -> Path:
    if path.suffix.lower() not in {".heic", ".heif"}:
        return path
    destination = path.with_suffix(f"{path.suffix}.jpg")
    if destination.exists() and destination.stat().st_mtime >= path.stat().st_mtime:
        return destination
    try:
        from PIL import Image
        import pillow_heif
    except ImportError:
        return _convert_heic_with_sips(path) or path
    pillow_heif.register_heif_opener()
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="JPEG", quality=95)
        return destination
    except Exception:
        return _convert_heic_with_sips(path) or path


def _convert_heic_with_sips(path: Path) -> Path | None:
    if not shutil.which("sips"):
        return None
    destination = path.with_suffix(f"{path.suffix}.jpg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return destination if destination.exists() else None


def _expires_at() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
