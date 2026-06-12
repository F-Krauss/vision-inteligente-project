from pathlib import Path

import pytest

from mold_inspection import piece_inspector


pytestmark = pytest.mark.skipif(
    piece_inspector.cv2 is None or piece_inspector.np is None,
    reason="OpenCV dependencies are not installed",
)


def test_reference_piece_diff_marks_small_localized_missing_region(tmp_path: Path):
    reference = _synthetic_fixture()
    candidate = reference.copy()
    piece_inspector.cv2.rectangle(candidate, (244, 178), (284, 218), (150, 150, 150), -1)
    reference_path = tmp_path / "reference.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    piece_inspector.cv2.imwrite(str(reference_path), reference)
    piece_inspector.cv2.imwrite(str(candidate_path), candidate)

    result = piece_inspector.inspect_expected_pieces_against_reference(
        "fam",
        "zona_01",
        candidate_path,
        reference_path,
        expected_pieces=[{"id": "broad_window", "class_name": "locator", "roi": [0.0, 0.0, 1.0, 1.0], "required": True}],
        evidence_dir=tmp_path / "evidence",
    )

    assert result["status"] == "review"
    assert result["missing_count"] == 1
    missing = [item for item in result["findings"] if item["status"] == "missing"]
    assert missing[0]["method"] == "reference_localized_diff"
    assert _polygon_area(missing[0]["region"]) < 0.04
    assert Path(result["overlay_image"]).exists()


def test_reference_piece_diff_matching_image_is_correct(tmp_path: Path):
    reference = _synthetic_fixture()
    reference_path = tmp_path / "reference.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    piece_inspector.cv2.imwrite(str(reference_path), reference)
    piece_inspector.cv2.imwrite(str(candidate_path), reference)

    result = piece_inspector.inspect_expected_pieces_against_reference(
        "fam",
        "zona_01",
        candidate_path,
        reference_path,
        expected_pieces=[{"id": "broad_window", "class_name": "locator", "roi": [0.0, 0.0, 1.0, 1.0], "required": True}],
    )

    assert result["status"] == "correct"
    assert result["missing_count"] == 0
    assert not [item for item in result["findings"] if item["status"] == "missing"]


def test_reference_piece_diff_global_change_is_review_without_missing_region(tmp_path: Path):
    reference = _synthetic_fixture()
    candidate = piece_inspector.np.clip(reference.astype("int16") + 55, 0, 255).astype("uint8")
    reference_path = tmp_path / "reference.jpg"
    candidate_path = tmp_path / "candidate.jpg"
    piece_inspector.cv2.imwrite(str(reference_path), reference)
    piece_inspector.cv2.imwrite(str(candidate_path), candidate)

    result = piece_inspector.inspect_expected_pieces_against_reference(
        "fam",
        "zona_01",
        candidate_path,
        reference_path,
        expected_pieces=[{"id": "broad_window", "class_name": "locator", "roi": [0.0, 0.0, 1.0, 1.0], "required": True}],
    )

    assert result["status"] == "review"
    assert result["missing_count"] == 0
    assert result["uncertain_count"] == 1


def _synthetic_fixture():
    cv2 = piece_inspector.cv2
    np = piece_inspector.np
    image = np.full((420, 620, 3), 150, dtype=np.uint8)
    cv2.rectangle(image, (70, 65), (550, 350), (95, 95, 95), -1)
    cv2.circle(image, (265, 198), 32, (35, 35, 35), -1)
    cv2.circle(image, (265, 198), 12, (210, 210, 210), -1)
    cv2.rectangle(image, (380, 150), (450, 220), (55, 55, 55), -1)
    return image


def _polygon_area(region: list[dict[str, float]]) -> float:
    xs = [point["x"] for point in region]
    ys = [point["y"] for point in region]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))
