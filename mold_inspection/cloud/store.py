from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
import json

from .config import CloudSettings
from .schemas import ResourceRecord, utc_now


COLLECTIONS = {
    "families",
    "molds",
    "zones",
    "datasets",
    "segmenter_datasets",
    "segmenter_training_jobs",
    "recipes",
    "inspector_training_jobs",
    "model_candidates",
    "public_dataset_imports",
    "model_versions",
    "inspections",
    "training_jobs",
    "uploads",
}


class MetadataStore:
    def list(self, collection: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get(self, collection: str, record_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def put(self, collection: str, record_id: str, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class LocalJsonStore(MetadataStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def list(self, collection: str) -> list[dict[str, Any]]:
        records = self._read_collection(collection)
        return sorted(records.values(), key=lambda item: item.get("created_at", ""))

    def get(self, collection: str, record_id: str) -> dict[str, Any] | None:
        return self._read_collection(collection).get(record_id)

    def put(self, collection: str, record_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            records = self._read_collection(collection)
            existing = records.get(record_id, {})
            now = utc_now()
            payload = {
                **existing,
                **data,
                "id": record_id,
                "created_at": existing.get("created_at", data.get("created_at", now)),
                "updated_at": now,
            }
            records[record_id] = payload
            self._path(collection).write_text(json.dumps(records, indent=2) + "\n")
            return payload

    def _read_collection(self, collection: str) -> dict[str, Any]:
        _validate_collection(collection)
        path = self._path(collection)
        if not path.exists():
            return {}
        return json.loads(path.read_text() or "{}")

    def _path(self, collection: str) -> Path:
        _validate_collection(collection)
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / f"{collection}.json"


class FirestoreStore(MetadataStore):
    def __init__(self, project_id: str):
        try:
            from google.cloud import firestore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install cloud dependencies to use Firestore metadata") from exc

        self.client = firestore.Client(project=project_id)

    def list(self, collection: str) -> list[dict[str, Any]]:
        _validate_collection(collection)
        return [doc.to_dict() | {"id": doc.id} for doc in self.client.collection(collection).stream()]

    def get(self, collection: str, record_id: str) -> dict[str, Any] | None:
        _validate_collection(collection)
        doc = self.client.collection(collection).document(record_id).get()
        if not doc.exists:
            return None
        return doc.to_dict() | {"id": doc.id}

    def put(self, collection: str, record_id: str, data: dict[str, Any]) -> dict[str, Any]:
        _validate_collection(collection)
        existing = self.get(collection, record_id) or {}
        now = utc_now()
        payload = {
            **existing,
            **data,
            "id": record_id,
            "created_at": existing.get("created_at", data.get("created_at", now)),
            "updated_at": now,
        }
        self.client.collection(collection).document(record_id).set(payload)
        return payload


def create_store(settings: CloudSettings) -> MetadataStore:
    if settings.metadata_backend == "firestore":
        return FirestoreStore(settings.project_id)
    return LocalJsonStore(settings.local_state_dir / "metadata")


def upsert_resource(store: MetadataStore, collection: str, data: dict[str, Any]) -> dict[str, Any]:
    record_id = str(data.get("id") or data.get("name") or data.get("slug") or "")
    if not record_id:
        raise ValueError("Resource payload must include id, name, or slug")
    record = ResourceRecord(id=record_id, data=data)
    return store.put(collection, record_id, record.model_dump())


def _validate_collection(collection: str) -> None:
    if collection not in COLLECTIONS:
        raise ValueError(f"Unknown collection: {collection}")
