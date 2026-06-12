from __future__ import annotations

import re
from typing import Any

from .schemas import MoldSectionPayload, MoldSectionPlanRecord, MoldSectionPlanUpsertRequest, utc_now
from .store import MetadataStore


def list_mold_section_plans(store: MetadataStore, family: str | None = None) -> list[dict[str, Any]]:
    records = store.list("mold_section_plans")
    if family:
        records = [record for record in records if str(record.get("family") or "") == family]
    return records


def get_mold_section_plan(store: MetadataStore, mold_key: str, family: str | None = None) -> dict[str, Any] | None:
    if family:
        record = store.get("mold_section_plans", section_plan_record_id(family, mold_key))
        if record:
            return record

    direct = store.get("mold_section_plans", mold_key)
    if direct:
        return direct

    for record in store.list("mold_section_plans"):
        if str(record.get("mold_key") or "") == mold_key and (family is None or str(record.get("family") or "") == family):
            return record
    return None


def upsert_mold_section_plan(
    store: MetadataStore,
    mold_key: str,
    request: MoldSectionPlanUpsertRequest,
) -> dict[str, Any]:
    family = request.family.strip()
    resolved_mold_key = (request.mold_key or request.mold_id or mold_key or family).strip()
    if not family:
        raise ValueError("family is required")
    if not resolved_mold_key:
        raise ValueError("mold_key is required")

    sections = _clean_sections(request.sections)
    if not sections:
        raise ValueError("At least one section is required")

    record_id = section_plan_record_id(family, resolved_mold_key)
    now = utc_now()
    record = MoldSectionPlanRecord(
        id=record_id,
        updated_at=now,
        family=family,
        mold_key=resolved_mold_key,
        mold_id=request.mold_id,
        name=request.name,
        source=request.source,
        sections=sections,
        section_count=len(sections),
        required_count=sum(1 for section in sections if section.required),
        suggestion_input=request.suggestion_input,
    )
    stored = store.put("mold_section_plans", record_id, record.model_dump(by_alias=False))
    _upsert_plan_zones(store, stored)
    return stored


def section_plan_record_id(family: str, mold_key: str) -> str:
    return f"{_slug(family)}__{_slug(mold_key or family)}"


def _clean_sections(sections: list[MoldSectionPayload]) -> list[MoldSectionPayload]:
    seen: set[str] = set()
    clean: list[MoldSectionPayload] = []
    for index, section in enumerate(sections, start=1):
        zone_id = section.zone_id.strip()
        section_id = section.id.strip() or f"section_{index:02d}_{section.view}"
        if not zone_id:
            raise ValueError("section zone_id is required")
        if section_id in seen:
            raise ValueError(f"Duplicate section id: {section_id}")
        seen.add(section_id)
        clean.append(
            MoldSectionPayload(
                id=section_id,
                zone_id=zone_id,
                label=section.label.strip() or f"Zona {section.zone_index} / {section.view}",
                zone_index=section.zone_index,
                view=section.view,
                required=section.required,
                order=section.order or index,
                notes=section.notes,
            )
        )
    return clean


def _upsert_plan_zones(store: MetadataStore, plan: dict[str, Any]) -> None:
    family = str(plan.get("family") or "")
    mold_key = str(plan.get("mold_key") or "")
    for section in plan.get("sections") or []:
        zone_id = str(section.get("zone_id") or "")
        if not zone_id:
            continue
        existing = store.get("zones", zone_id) or {}
        store.put(
            "zones",
            zone_id,
            {
                **existing,
                "id": zone_id,
                "family": family,
                "mold_key": mold_key,
                "name": section.get("label") or zone_id,
                "section_plan_id": plan.get("id"),
                "zone_index": section.get("zone_index"),
                "view": section.get("view"),
                "required": section.get("required", True),
            },
        )


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    return slug or "default"
