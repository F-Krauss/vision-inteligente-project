from pathlib import Path

import pytest

from mold_inspection import anomaly_reference


pytestmark = pytest.mark.skipif(
    anomaly_reference.cv2 is None,
    reason="OpenCV dependencies are not installed",
)


def test_anomaly_ignores_change_outside_mask(tmp_path: Path):
    image = _synthetic_zone()
    changed = image.copy()
    changed[:, :70] = (0, 0, 255)
    mask = _synthetic_mask(image.shape[:2])
    paths = _write_case(tmp_path, image, changed, mask)

    anomaly_reference.train_anomaly_model(
        family="fam",
        zone_id="zona_01",
        images=[paths["reference"]],
        mask_path=paths["mask"],
        anomaly_dir=tmp_path / "anomaly",
        feature_backend="classical",
        image_size=128,
        anomaly_threshold=0.08,
        heatmap_threshold=0.08,
        min_region_area=80,
    )
    reports = anomaly_reference.inspect_anomaly_images(
        family="fam",
        zone_id="zona_01",
        images=[paths["candidate"]],
        anomaly_dir=tmp_path / "anomaly",
        evidence_dir=tmp_path / "evidence",
    )

    assert reports[0]["result"]["status"] == "correct"
    assert not reports[0]["result"]["difference_regions"]


def test_anomaly_marks_change_inside_mask(tmp_path: Path):
    image = _synthetic_zone()
    changed = image.copy()
    anomaly_reference.cv2.rectangle(changed, (185, 130), (255, 190), (170, 170, 170), -1)
    mask = _synthetic_mask(image.shape[:2])
    paths = _write_case(tmp_path, image, changed, mask)

    anomaly_reference.train_anomaly_model(
        family="fam",
        zone_id="zona_01",
        images=[paths["reference"]],
        mask_path=paths["mask"],
        anomaly_dir=tmp_path / "anomaly",
        feature_backend="classical",
        image_size=128,
        anomaly_threshold=0.05,
        heatmap_threshold=0.05,
        min_region_area=50,
    )
    reports = anomaly_reference.inspect_anomaly_images(
        family="fam",
        zone_id="zona_01",
        images=[paths["candidate"]],
        anomaly_dir=tmp_path / "anomaly",
        evidence_dir=tmp_path / "evidence",
    )

    result = reports[0]["result"]
    assert result["status"] == "review"
    assert result["difference_regions"]
    assert Path(result["overlay_image"]).exists()


def test_anomaly_requests_retake_for_unalignable_image(tmp_path: Path):
    image = _synthetic_zone()
    blank = anomaly_reference.np.full_like(image, 160)
    mask = _synthetic_mask(image.shape[:2])
    paths = _write_case(tmp_path, image, blank, mask)

    anomaly_reference.train_anomaly_model(
        family="fam",
        zone_id="zona_01",
        images=[paths["reference"]],
        mask_path=paths["mask"],
        anomaly_dir=tmp_path / "anomaly",
        feature_backend="classical",
        image_size=128,
    )
    reports = anomaly_reference.inspect_anomaly_images(
        family="fam",
        zone_id="zona_01",
        images=[paths["candidate"]],
        anomaly_dir=tmp_path / "anomaly",
        evidence_dir=tmp_path / "evidence",
    )

    assert reports[0]["result"]["status"] == "retake_photo"


def _write_case(tmp_path: Path, reference, candidate, mask) -> dict[str, str]:
    reference_path = tmp_path / "reference.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    mask_path = tmp_path / "mask.png"
    anomaly_reference.cv2.imwrite(str(reference_path), reference)
    anomaly_reference.cv2.imwrite(str(candidate_path), candidate)
    anomaly_reference.cv2.imwrite(str(mask_path), mask)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "mask": str(mask_path),
    }


def _synthetic_zone():
    cv2 = anomaly_reference.cv2
    np = anomaly_reference.np
    image = np.full((320, 420, 3), 170, dtype=np.uint8)
    cv2.rectangle(image, (95, 70), (345, 250), (120, 120, 120), -1)
    cv2.rectangle(image, (115, 95), (165, 150), (55, 55, 55), -1)
    cv2.rectangle(image, (185, 130), (255, 190), (40, 40, 40), -1)
    cv2.circle(image, (300, 120), 25, (35, 35, 35), -1)
    for x in range(120, 330, 40):
        cv2.circle(image, (x, 220), 9, (20, 20, 20), -1)
        cv2.circle(image, (x, 220), 4, (220, 220, 220), -1)
    return image


def _synthetic_mask(shape: tuple[int, int]):
    np = anomaly_reference.np
    cv2 = anomaly_reference.cv2
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.rectangle(mask, (90, 65), (350, 255), 255, -1)
    return mask
