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


def _det(index, class_name, conf, bbox):
    return {"det_index": index, "class_name": class_name, "confidence": conf, "bbox": bbox}


def test_assignment_flags_specific_missing_screw_among_identical_parts():
    # Three identical screws expected in three distinct slots; only the right slot
    # actually has a screw. Class-only matching would mark all three present.
    required = [
        {"id": "screw_left", "class_name": "screw", "roi": [0.0, 0.0, 0.33, 1.0]},
        {"id": "screw_mid", "class_name": "screw", "roi": [0.33, 0.0, 0.66, 1.0]},
        {"id": "screw_right", "class_name": "screw", "roi": [0.66, 0.0, 1.0, 1.0]},
    ]
    detections = [_det(0, "screw", 0.9, [880, 40, 920, 80])]  # only inside the right slot (w=1000)
    assigned, consumed = piece_inspector._assign_detections_to_pieces(detections, required, 1000, 100)

    assert set(assigned) == {"screw_right"}
    assert "screw_left" not in assigned and "screw_mid" not in assigned
    assert consumed == {0}


def test_assignment_consumes_each_detection_once():
    # One detection cannot satisfy two overlapping expected slots.
    required = [
        {"id": "a", "class_name": "screw", "roi": [0.0, 0.0, 0.6, 1.0]},
        {"id": "b", "class_name": "screw", "roi": [0.4, 0.0, 1.0, 1.0]},
    ]
    detections = [_det(0, "screw", 0.9, [480, 40, 520, 80])]  # in the overlap band
    assigned, consumed = piece_inspector._assign_detections_to_pieces(detections, required, 1000, 100)

    assert len(assigned) == 1 and len(consumed) == 1


def _textured_image(w=960, h=720):
    np = piece_inspector.np
    cv2 = piece_inspector.cv2
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    for _ in range(600):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, 14))
        c = tuple(int(v) for v in rng.integers(60, 240, size=3))
        cv2.circle(img, (x, y), r, c, -1)
    for _ in range(120):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, (200, 200, 200), 1)
    return img


def _control_point_error(h_est, h_true, w, h, n=12):
    np = piece_inspector.np
    cv2 = piece_inspector.cv2
    xs = np.linspace(w * 0.2, w * 0.8, n)
    ys = np.linspace(h * 0.2, h * 0.8, n)
    grid = np.float32([[[x, y]] for x in xs for y in ys])
    a = cv2.perspectiveTransform(grid, h_est.astype(np.float64)).reshape(-1, 2)
    b = cv2.perspectiveTransform(grid, h_true.astype(np.float64)).reshape(-1, 2)
    return float(piece_inspector.np.linalg.norm(a - b, axis=1).max())


def test_ecc_refinement_reaches_subpixel_alignment():
    np = piece_inspector.np
    cv2 = piece_inspector.cv2
    ref = _textured_image()
    h, w = ref.shape[:2]
    # candidate→reference ground-truth homography (mild projective + translation).
    h_true = np.array([[1.0, 0.02, 9.0], [0.012, 1.0, -7.0], [1.2e-5, 6e-6, 1.0]])
    candidate = cv2.warpPerspective(ref, np.linalg.inv(h_true), (w, h))
    # lighting (gain+bias) + sensor noise that degrades pure feature localization.
    candidate = np.clip(candidate.astype(np.float32) * 0.82 + 22, 0, 255).astype(np.uint8)
    candidate = np.clip(candidate.astype(np.int16) + np.random.default_rng(3).integers(-6, 7, candidate.shape), 0, 255).astype(np.uint8)

    _, alignment = piece_inspector._align_candidate_to_reference(ref, candidate)
    assert alignment["ok"]
    assert alignment["method"] == "orb+ecc"
    assert _control_point_error(alignment["_homography"], h_true, w, h) < 1.5


def test_ecc_refinement_improves_a_perturbed_homography():
    np = piece_inspector.np
    cv2 = piece_inspector.cv2
    ref = _textured_image()
    h, w = ref.shape[:2]
    h_true = np.array([[1.0, 0.015, 6.0], [0.01, 1.0, -5.0], [1e-5, 5e-6, 1.0]])
    candidate = cv2.warpPerspective(ref, np.linalg.inv(h_true), (w, h))
    candidate = np.clip(candidate.astype(np.float32) * 0.85 + 18, 0, 255).astype(np.uint8)

    # Simulate an ORB homography that is a few px off the truth.
    h_init = h_true.copy()
    h_init[0, 2] += 4.5
    h_init[1, 2] -= 3.5
    err_before = _control_point_error(h_init, h_true, w, h)

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    refined, info = piece_inspector._refine_homography_ecc(ref_gray, cand_gray, h_init)
    err_after = _control_point_error(refined, h_true, w, h)

    assert info["applied"]
    assert err_after < err_before
    assert err_after < 1.5


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
