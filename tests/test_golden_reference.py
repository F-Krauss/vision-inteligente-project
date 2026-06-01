from pathlib import Path

import pytest

from mold_inspection import golden_reference


pytestmark = pytest.mark.skipif(
    golden_reference.cv2 is None,
    reason="OpenCV dependencies are not installed",
)


def test_golden_reference_detects_matching_image(tmp_path: Path):
    image = _synthetic_mold_image()
    reference_path = tmp_path / "ref.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    golden_reference.cv2.imwrite(str(reference_path), image)
    golden_reference.cv2.imwrite(str(candidate_path), image)

    created = golden_reference.create_reference(
        source_image=reference_path,
        family="fam",
        zone_id="zona_01",
        references_dir=tmp_path / "references",
    )
    assert created.exists()

    result = golden_reference.inspect_against_reference(
        image_path=candidate_path,
        family="fam",
        zone_id="zona_01",
        references_dir=tmp_path / "references",
        evidence_dir=tmp_path / "evidence",
        min_similarity=0.98,
        min_keypoints=8,
        min_inlier_ratio=0.10,
        min_region_area=200,
    )

    assert result.status == "correct"
    assert result.quality.ok
    assert result.similarity is not None
    assert result.matched_reference == "default"


def test_golden_reference_detects_changed_region(tmp_path: Path):
    image = _synthetic_mold_image()
    changed = image.copy()
    golden_reference.cv2.rectangle(changed, (220, 180), (280, 240), (0, 0, 0), -1)
    reference_path = tmp_path / "ref.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    golden_reference.cv2.imwrite(str(reference_path), image)
    golden_reference.cv2.imwrite(str(candidate_path), changed)
    golden_reference.create_reference(reference_path, "fam", "zona_01", references_dir=tmp_path / "references")

    result = golden_reference.inspect_against_reference(
        image_path=candidate_path,
        family="fam",
        zone_id="zona_01",
        references_dir=tmp_path / "references",
        evidence_dir=tmp_path / "evidence",
        min_similarity=0.99,
        min_keypoints=8,
        min_inlier_ratio=0.10,
        min_region_area=100,
    )

    assert result.status == "review"
    assert result.difference_regions


def test_golden_reference_selects_best_reference_variant(tmp_path: Path):
    image = _synthetic_mold_image()
    alternate = golden_reference.cv2.rotate(image, golden_reference.cv2.ROTATE_180)
    candidate_path = tmp_path / "candidate.jpg"
    reference_a_path = tmp_path / "front.jpg"
    reference_b_path = tmp_path / "side.jpg"
    golden_reference.cv2.imwrite(str(candidate_path), alternate)
    golden_reference.cv2.imwrite(str(reference_a_path), image)
    golden_reference.cv2.imwrite(str(reference_b_path), alternate)

    golden_reference.create_reference(reference_a_path, "fam", "lateral_pistones_01", "frontal", tmp_path / "references")
    golden_reference.create_reference(reference_b_path, "fam", "lateral_pistones_01", "lateral", tmp_path / "references")

    result = golden_reference.inspect_against_reference(
        image_path=candidate_path,
        family="fam",
        zone_id="lateral_pistones_01",
        references_dir=tmp_path / "references",
        evidence_dir=tmp_path / "evidence",
        min_similarity=0.98,
        min_keypoints=8,
        min_inlier_ratio=0.10,
        min_region_area=200,
    )

    assert result.status == "correct"
    assert result.matched_reference == "lateral"


def test_similarity_ignores_invalid_warp_border():
    image = _synthetic_mold_image()
    aligned = image.copy()
    aligned[:, :80] = 0
    valid_mask = golden_reference.np.full(image.shape[:2], 255, dtype=golden_reference.np.uint8)
    valid_mask[:, :90] = 0

    score, diff_map = golden_reference._similarity(image, aligned, valid_mask)
    regions, _ = golden_reference._difference_regions(diff_map, 38, 100, valid_mask)

    assert score > 0.99
    assert not regions


def test_large_difference_area_ratio_requests_retake(tmp_path: Path):
    reference = _synthetic_mold_image()
    candidate = reference.copy()
    candidate[:, :] = 35
    reference_path = tmp_path / "ref.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    golden_reference.cv2.imwrite(str(reference_path), reference)
    golden_reference.cv2.imwrite(str(candidate_path), candidate)
    golden_reference.create_reference(reference_path, "fam", "zona_01", references_dir=tmp_path / "references")

    result = golden_reference.inspect_against_reference(
        image_path=candidate_path,
        family="fam",
        zone_id="zona_01",
        references_dir=tmp_path / "references",
        evidence_dir=tmp_path / "evidence",
        min_keypoints=1,
        min_inlier_ratio=0.0,
        min_blur_score=0.0,
        max_brightness_delta=255.0,
    )

    assert result.status == "retake_photo"


def _synthetic_mold_image():
    cv2 = golden_reference.cv2
    np = golden_reference.np
    image = np.full((420, 620, 3), 170, dtype=np.uint8)
    for x in range(60, 560, 90):
        cv2.circle(image, (x, 80), 18, (35, 35, 35), -1)
        cv2.circle(image, (x, 80), 8, (215, 215, 215), -1)
    for index, x in enumerate(range(80, 520, 110)):
        y = 180 + (index % 2) * 60
        cv2.rectangle(image, (x, y), (x + 70, y + 35), (75, 75, 75), -1)
        cv2.rectangle(image, (x + 8, y + 8), (x + 62, y + 27), (210, 210, 210), 2)
    for y in range(300, 390, 35):
        cv2.line(image, (70, y), (540, y), (60, 60, 60), 3)
    return image
