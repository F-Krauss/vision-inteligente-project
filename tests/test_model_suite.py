from pathlib import Path

from mold_inspection import cli
from mold_inspection import anomaly_reference
from mold_inspection.model_suite import export_best_model, inspect_best_model, train_model_suite


def test_train_model_suite_exports_best_and_inspects(tmp_path: Path):
    if anomaly_reference.cv2 is None:
        return

    paths = _write_supervised_dataset(tmp_path)
    report = train_model_suite(
        family="fam",
        zone_id="zona_01",
        manifest_path=paths["manifest"],
        mask_path=paths["mask"],
        registry_dir=tmp_path / "registry",
        evidence_dir=tmp_path / "suite_evidence",
    )

    best_dir = tmp_path / "registry" / "fam" / "zona_01" / "best_model"
    assert report["selected_candidate"].startswith("patchcore_classical")
    assert (best_dir / "model.npz").exists()
    assert (best_dir / "thresholds.json").exists()
    assert (best_dir / "benchmark.json").exists()

    reports = inspect_best_model(
        family="fam",
        zone_id="zona_01",
        images=[paths["fault"]],
        registry_dir=tmp_path / "registry",
        evidence_dir=tmp_path / "best_evidence",
    )

    assert reports[0]["result"]["model_id"] == report["selected_candidate"]
    assert reports[0]["result"]["status"] in {"review", "retake_photo"}
    assert reports[0]["result"]["overlay_image"] or reports[0]["result"]["status"] == "retake_photo"


def test_model_suite_cli_commands(tmp_path: Path):
    if anomaly_reference.cv2 is None:
        return

    paths = _write_supervised_dataset(tmp_path)
    train_result = cli.main(
        [
            "train-model-suite",
            "--family",
            "fam",
            "--zone-id",
            "zona_01",
            "--manifest",
            paths["manifest"],
            "--mask",
            paths["mask"],
            "--registry-dir",
            str(tmp_path / "registry"),
            "--evidence-dir",
            str(tmp_path / "suite_evidence"),
        ]
    )
    export_result = cli.main(
        [
            "export-best",
            "--family",
            "fam",
            "--zone-id",
            "zona_01",
            "--registry-dir",
            str(tmp_path / "registry"),
        ]
    )
    inspect_result = cli.main(
        [
            "inspect-best",
            "--family",
            "fam",
            "--zone-id",
            "zona_01",
            "--images",
            paths["ok_val"],
            "--registry-dir",
            str(tmp_path / "registry"),
            "--evidence-dir",
            str(tmp_path / "best_evidence"),
            "--out",
            str(tmp_path / "best_report.json"),
        ]
    )

    assert train_result == 0
    assert export_result == 0
    assert inspect_result == 0
    assert (tmp_path / "best_report.json").exists()


def _write_supervised_dataset(tmp_path: Path) -> dict[str, str]:
    cv2 = anomaly_reference.cv2
    np = anomaly_reference.np
    ok = np.full((260, 340, 3), 170, dtype=np.uint8)
    cv2.rectangle(ok, (80, 65), (270, 205), (105, 105, 105), -1)
    cv2.rectangle(ok, (110, 95), (155, 145), (45, 45, 45), -1)
    cv2.circle(ok, (220, 125), 24, (35, 35, 35), -1)
    for x in range(110, 260, 35):
        cv2.circle(ok, (x, 180), 8, (20, 20, 20), -1)
    ok_val = ok.copy()
    fault = ok.copy()
    cv2.rectangle(fault, (110, 95), (155, 145), (165, 165, 165), -1)
    mask = np.zeros(ok.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (70, 55), (280, 215), 255, -1)

    ok_train_path = tmp_path / "ok_train.jpg"
    ok_val_path = tmp_path / "ok_val.jpg"
    fault_path = tmp_path / "fault.jpg"
    mask_path = tmp_path / "mask.png"
    manifest_path = tmp_path / "manifest.csv"
    cv2.imwrite(str(ok_train_path), ok)
    cv2.imwrite(str(ok_val_path), ok_val)
    cv2.imwrite(str(fault_path), fault)
    cv2.imwrite(str(mask_path), mask)
    manifest_path.write_text(
        "image_path,family,zone_id,label,mold_id,session_id,split\n"
        f"{ok_train_path},fam,zona_01,ok,molde_1,sesion_train,train\n"
        f"{ok_val_path},fam,zona_01,ok,molde_2,sesion_val,val\n"
        f"{fault_path},fam,zona_01,fault,molde_3,sesion_val,val\n"
    )
    return {
        "manifest": str(manifest_path),
        "mask": str(mask_path),
        "ok_train": str(ok_train_path),
        "ok_val": str(ok_val_path),
        "fault": str(fault_path),
    }
