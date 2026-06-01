from mold_inspection.decision import inspect_zone
from mold_inspection.models import Box, Detection, ElementStatus, ExpectedElement, ZoneConfig, ZoneStatus


def test_present_detection_inside_roi_marks_zone_correct():
    zone = ZoneConfig(
        id="zona_01",
        expected=[
            ExpectedElement(
                id="tornillo_01",
                class_name="screw",
                roi=Box(0.1, 0.1, 0.3, 0.3),
                min_confidence=0.5,
                min_overlap=0.2,
            )
        ],
    )

    result = inspect_zone(
        zone,
        [Detection(class_name="screw", confidence=0.8, bbox=Box(0.12, 0.12, 0.25, 0.25))],
    )

    assert result.status == ZoneStatus.CORRECT
    assert result.findings[0].status == ElementStatus.PRESENT


def test_low_confidence_detection_marks_review():
    zone = ZoneConfig(
        id="zona_01",
        expected=[
            ExpectedElement(
                id="tornillo_01",
                class_name="screw",
                roi=Box(0.1, 0.1, 0.3, 0.3),
                min_confidence=0.8,
                min_overlap=0.2,
            )
        ],
    )

    result = inspect_zone(
        zone,
        [Detection(class_name="screw", confidence=0.55, bbox=Box(0.12, 0.12, 0.25, 0.25))],
    )

    assert result.status == ZoneStatus.REVIEW
    assert result.findings[0].status == ElementStatus.DOUBTFUL


def test_missing_critical_element_marks_incorrect():
    zone = ZoneConfig(
        id="zona_01",
        expected=[ExpectedElement(id="bloque_01", class_name="block", critical=True)],
    )

    result = inspect_zone(zone, [])

    assert result.status == ZoneStatus.INCORRECT
    assert result.findings[0].status == ElementStatus.ABSENT
