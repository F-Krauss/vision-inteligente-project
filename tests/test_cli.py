from pathlib import Path

from mold_inspection import cli, golden_reference


def test_set_references_cli_creates_numbered_reference_files(tmp_path: Path):
    if golden_reference.cv2 is None:
        return

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    image = golden_reference.np.full((120, 160, 3), 180, dtype=golden_reference.np.uint8)
    golden_reference.cv2.circle(image, (70, 50), 20, (30, 30, 30), -1)
    golden_reference.cv2.imwrite(str(first), image)
    golden_reference.cv2.imwrite(str(second), image)

    result = cli.main(
        [
            "set-references",
            "--images",
            str(first),
            str(second),
            "--family",
            "fam",
            "--zone-id",
            "frontal_zona_01",
            "--reference-prefix",
            "frontal",
            "--references-dir",
            str(tmp_path / "references"),
        ]
    )

    assert result == 0
    assert (tmp_path / "references" / "fam" / "frontal_zona_01" / "frontal_001.jpg").exists()
    assert (tmp_path / "references" / "fam" / "frontal_zona_01" / "frontal_002.jpg").exists()


def test_anomaly_cli_trains_and_inspects(tmp_path: Path):
    if golden_reference.cv2 is None:
        return

    image = golden_reference.np.full((180, 220, 3), 170, dtype=golden_reference.np.uint8)
    golden_reference.cv2.rectangle(image, (55, 45), (165, 135), (80, 80, 80), -1)
    golden_reference.cv2.circle(image, (105, 90), 20, (25, 25, 25), -1)
    mask = golden_reference.np.zeros(image.shape[:2], dtype=golden_reference.np.uint8)
    golden_reference.cv2.rectangle(mask, (45, 35), (175, 145), 255, -1)
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "mask.png"
    report_path = tmp_path / "report.json"
    golden_reference.cv2.imwrite(str(image_path), image)
    golden_reference.cv2.imwrite(str(mask_path), mask)

    train_result = cli.main(
        [
            "train-anomaly",
            "--family",
            "fam",
            "--zone-id",
            "zona_01",
            "--images",
            str(image_path),
            "--mask",
            str(mask_path),
            "--anomaly-dir",
            str(tmp_path / "anomaly"),
            "--feature-backend",
            "classical",
            "--image-size",
            "128",
        ]
    )
    inspect_result = cli.main(
        [
            "inspect-anomaly",
            "--family",
            "fam",
            "--zone-id",
            "zona_01",
            "--images",
            str(image_path),
            "--anomaly-dir",
            str(tmp_path / "anomaly"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--out",
            str(report_path),
        ]
    )

    assert train_result == 0
    assert inspect_result == 0
    assert (tmp_path / "anomaly" / "fam" / "zona_01" / "profile.json").exists()
    assert (tmp_path / "anomaly" / "fam" / "zona_01" / "memory_bank.npz").exists()
    assert report_path.exists()
