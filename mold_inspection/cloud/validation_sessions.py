from __future__ import annotations

from typing import Any

from .schemas import (
    MoldSectionPayload,
    MoldValidationSectionUpdateRequest,
    MoldValidationSessionCreateRequest,
    MoldValidationSessionRecord,
    utc_now,
)
from .section_plans import get_mold_section_plan
from .store import MetadataStore


def list_mold_validation_sessions(
    store: MetadataStore,
    family: str | None = None,
    mold_key: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    records = store.list("mold_validation_sessions")
    if family:
        records = [record for record in records if str(record.get("family") or "") == family]
    if mold_key:
        records = [record for record in records if str(record.get("mold_key") or "") == mold_key]
    if status:
        records = [record for record in records if str(record.get("status") or "") == status]
    return records


def create_mold_validation_session(store: MetadataStore, request: MoldValidationSessionCreateRequest) -> dict[str, Any]:
    family = request.family.strip()
    mold_key = (request.mold_key or request.mold_id or family).strip()
    if not family:
        raise ValueError("family is required")
    if not mold_key:
        raise ValueError("mold_key is required")

    plan = _load_plan(store, family, mold_key, request.plan_id)
    required_sections = _required_sections(plan)
    if not required_sections:
        raise ValueError("Section plan has no required sections")

    for record in reversed(store.list("mold_validation_sessions")):
        if (
            str(record.get("family") or "") == family
            and str(record.get("mold_key") or "") == mold_key
            and str(record.get("status") or "") != "complete"
        ):
            refreshed = _recompute_session_status(
                {
                    **record,
                    "plan_id": str(plan["id"]),
                    "required_sections": [section.model_dump(by_alias=False) for section in required_sections],
                    "updated_at": utc_now(),
                },
                required_sections,
            )
            return store.put("mold_validation_sessions", str(record["id"]), refreshed)

    session = MoldValidationSessionRecord(
        family=family,
        mold_key=mold_key,
        mold_id=request.mold_id,
        operator_id=request.operator_id,
        plan_id=str(plan["id"]),
        required_sections=required_sections,
        required_count=len(required_sections),
        missing_section_ids=[section.id for section in required_sections],
    )
    return store.put("mold_validation_sessions", session.id, session.model_dump(by_alias=False))


def get_mold_validation_session(store: MetadataStore, session_id: str) -> dict[str, Any] | None:
    return store.get("mold_validation_sessions", session_id)


def update_mold_validation_section(
    store: MetadataStore,
    session_id: str,
    request: MoldValidationSectionUpdateRequest,
) -> dict[str, Any]:
    session = store.get("mold_validation_sessions", session_id)
    if not session:
        raise ValueError("Validation session not found")

    sections = _sections_from_session(session)
    section = _match_section(sections, request.section_id, request.zone_id)
    if not section:
        raise ValueError("Section is not part of this validation session")

    now = utc_now()
    section_results = dict(session.get("section_results") or {})
    section_results[section.id] = {
        "section_id": section.id,
        "zone_id": section.zone_id,
        "label": section.label,
        "status": request.status,
        "inspection_id": request.inspection_id,
        "image_uri": request.image_uri,
        "message": request.message,
        "reviewed_by": request.reviewed_by,
        "notes": request.notes,
        "updated_at": now,
    }
    payload = _recompute_session_status({**session, "section_results": section_results, "updated_at": now}, sections)
    return store.put("mold_validation_sessions", session_id, payload)


def _load_plan(store: MetadataStore, family: str, mold_key: str, plan_id: str | None) -> dict[str, Any]:
    plan = store.get("mold_section_plans", plan_id) if plan_id else None
    if not plan:
        plan = get_mold_section_plan(store, mold_key, family=family)
    if not plan:
        raise ValueError("Section plan not found")
    return plan


def _required_sections(plan: dict[str, Any]) -> list[MoldSectionPayload]:
    sections: list[MoldSectionPayload] = []
    for section in plan.get("sections") or []:
        item = MoldSectionPayload.model_validate(section)
        if item.required:
            sections.append(item)
    return sections


def _sections_from_session(session: dict[str, Any]) -> list[MoldSectionPayload]:
    return [MoldSectionPayload.model_validate(section) for section in session.get("required_sections") or []]


def _match_section(
    sections: list[MoldSectionPayload],
    section_id: str | None,
    zone_id: str | None,
) -> MoldSectionPayload | None:
    for section in sections:
        if section_id and section.id == section_id:
            return section
        if zone_id and section.zone_id == zone_id:
            return section
    return None


def _recompute_session_status(session: dict[str, Any], sections: list[MoldSectionPayload]) -> dict[str, Any]:
    section_results = session.get("section_results") or {}
    ready_ids = [
        section.id
        for section in sections
        if (section_results.get(section.id) or {}).get("status") in {"correct", "review"}
    ]
    missing_ids = [section.id for section in sections if section.id not in set(ready_ids)]
    required_count = len(sections)
    completed_count = len(ready_ids)
    status = "complete" if required_count and completed_count == required_count else "in_progress" if section_results else "pending"
    completed_at = session.get("completed_at")
    if status == "complete" and not completed_at:
        completed_at = utc_now()
    if status != "complete":
        completed_at = None
    return {
        **session,
        "status": status,
        "required_count": required_count,
        "completed_count": completed_count,
        "ready_section_ids": ready_ids,
        "missing_section_ids": missing_ids,
        "completed_at": completed_at,
    }
