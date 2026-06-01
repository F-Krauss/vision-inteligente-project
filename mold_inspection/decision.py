from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import Detection, ElementFinding, ElementStatus, ExpectedElement, ZoneConfig, ZoneStatus


@dataclass(frozen=True)
class ZoneInspectionResult:
    zone_id: str
    status: ZoneStatus
    findings: list[ElementFinding]
    extra_detections: list[Detection]

    def as_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "status": self.status.value,
            "findings": [finding.as_dict() for finding in self.findings],
            "extra_detections": [detection.as_dict() for detection in self.extra_detections],
        }


def inspect_zone(zone: ZoneConfig, detections: Iterable[Detection]) -> ZoneInspectionResult:
    detections = list(detections)
    findings = [_find_element(element, detections) for element in zone.expected]
    extra_detections = _extra_detections(zone, detections)

    if any(f.status == ElementStatus.ABSENT and f.critical for f in findings):
        status = ZoneStatus.INCORRECT
    elif any(f.status == ElementStatus.DOUBTFUL for f in findings) or extra_detections:
        status = ZoneStatus.REVIEW
    else:
        status = ZoneStatus.CORRECT

    return ZoneInspectionResult(
        zone_id=zone.id,
        status=status,
        findings=findings,
        extra_detections=extra_detections,
    )


def _find_element(element: ExpectedElement, detections: list[Detection]) -> ElementFinding:
    candidates = [detection for detection in detections if detection.class_name == element.class_name]
    candidates.sort(key=lambda item: item.confidence, reverse=True)

    best_present = None
    best_doubtful = None
    doubtful_confidence = element.min_confidence * 0.6

    for detection in candidates:
        overlap_ok = _overlap_ok(element, detection, strict=True)
        weak_overlap_ok = _overlap_ok(element, detection, strict=False)

        if detection.confidence >= element.min_confidence and overlap_ok:
            best_present = detection
            break

        if detection.confidence >= doubtful_confidence and weak_overlap_ok and best_doubtful is None:
            best_doubtful = detection

    if best_present:
        return ElementFinding(
            element_id=element.id,
            class_name=element.class_name,
            status=ElementStatus.PRESENT,
            critical=element.critical,
            confidence=best_present.confidence,
            bbox=best_present.bbox,
        )

    if best_doubtful:
        return ElementFinding(
            element_id=element.id,
            class_name=element.class_name,
            status=ElementStatus.DOUBTFUL,
            critical=element.critical,
            confidence=best_doubtful.confidence,
            bbox=best_doubtful.bbox,
        )

    return ElementFinding(
        element_id=element.id,
        class_name=element.class_name,
        status=ElementStatus.ABSENT,
        critical=element.critical,
    )


def _overlap_ok(element: ExpectedElement, detection: Detection, strict: bool) -> bool:
    if element.roi is None:
        return True
    if detection.bbox is None:
        return False

    threshold = element.min_overlap if strict else max(0.05, element.min_overlap * 0.5)
    return detection.bbox.overlap_ratio(element.roi) >= threshold


def _extra_detections(zone: ZoneConfig, detections: list[Detection]) -> list[Detection]:
    expected_classes = {element.class_name for element in zone.expected}
    return [
        detection
        for detection in detections
        if detection.class_name not in expected_classes and detection.confidence >= 0.4
    ]
