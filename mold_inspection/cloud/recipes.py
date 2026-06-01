from __future__ import annotations

from typing import Any

from .schemas import RecipeCreateRequest, RecipeRecord
from .store import MetadataStore


def create_recipe(request: RecipeCreateRequest, store: MetadataStore) -> RecipeRecord:
    readiness = _readiness(request.family, request.zone_id, store)
    record = RecipeRecord(
        family=request.family,
        zone_id=request.zone_id,
        name=request.name,
        mold_id=request.mold_id,
        objective=request.objective,
        status=_recipe_status(readiness),
        readiness=readiness,
        notes=request.notes,
    )
    store.put("recipes", record.id, record.model_dump())
    return record


def list_recipes(store: MetadataStore) -> list[dict[str, Any]]:
    recipes = store.list("recipes")
    updated: list[dict[str, Any]] = []
    for recipe in recipes:
        readiness = _readiness(str(recipe.get("family")), str(recipe.get("zone_id")), store)
        candidate = _best_candidate(recipe["family"], recipe["zone_id"], store)
        payload = {
            **recipe,
            "readiness": readiness,
            "status": _recipe_status(readiness, candidate),
            "best_model_candidate_id": candidate.get("id") if candidate else recipe.get("best_model_candidate_id"),
            "metrics": candidate.get("metrics", {}) if candidate else recipe.get("metrics", {}),
        }
        store.put("recipes", recipe["id"], payload)
        updated.append(payload)
    return updated


def _readiness(family: str, zone_id: str, store: MetadataStore) -> dict[str, Any]:
    dataset = _latest_matching(store.list("datasets"), family, zone_id)
    segmenter_job = _latest_matching(store.list("segmenter_training_jobs"), None, None)
    inspector_job = _latest_matching(store.list("inspector_training_jobs"), family, zone_id)
    candidate = _best_candidate(family, zone_id, store)
    return {
        "dataset_ready": bool(dataset),
        "dataset_id": dataset.get("id") if dataset else None,
        "ok_count": dataset.get("ok_count", 0) if dataset else 0,
        "fault_count": dataset.get("fault_count", 0) if dataset else 0,
        "piece_count": dataset.get("piece_count", 0) if dataset else 0,
        "mask_ready": bool(dataset and dataset.get("mask_uri")),
        "manifest_ready": bool(dataset and dataset.get("manifest_uri")),
        "segmenter_status": segmenter_job.get("status") if segmenter_job else "not_started",
        "inspector_status": inspector_job.get("status") if inspector_job else "not_started",
        "best_model_ready": bool(candidate),
    }


def _recipe_status(readiness: dict[str, Any], candidate: dict[str, Any] | None = None) -> str:
    if candidate or readiness.get("best_model_ready"):
        return "ready_for_inspection"
    if readiness.get("inspector_status") in {"queued", "submitted", "running"}:
        return "training"
    if readiness.get("dataset_ready"):
        return "ready_for_training"
    return "draft"


def _latest_matching(records: list[dict[str, Any]], family: str | None, zone_id: str | None) -> dict[str, Any] | None:
    for record in reversed(records):
        source = record.get("data") if isinstance(record.get("data"), dict) else record
        if family is not None and source.get("family") != family:
            continue
        if zone_id is not None and source.get("zone_id") != zone_id:
            continue
        return source
    return None


def _best_candidate(family: str, zone_id: str, store: MetadataStore) -> dict[str, Any] | None:
    candidates = []
    for record in store.list("model_candidates"):
        source = record.get("data") if isinstance(record.get("data"), dict) else record
        if source.get("family") == family and source.get("zone_id") == zone_id and source.get("promoted"):
            candidates.append(source)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.get("promoted_at", item.get("created_at", "")))[-1]
