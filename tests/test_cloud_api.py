from pathlib import Path
import csv
import json

from fastapi.testclient import TestClient

from mold_inspection.cloud.app import create_app
from mold_inspection.cloud.config import CloudSettings
from mold_inspection.cloud.store import _decode_firestore_nested_arrays, _encode_firestore_nested_arrays


def test_cloud_api_upload_and_review_without_model(tmp_path: Path):
    image_path = tmp_path / "zone.jpg"
    _write_minimal_jpeg(image_path)
    settings = CloudSettings(
        local_state_dir=tmp_path / "state",
        model_registry_dir=tmp_path / "registry",
        evidence_dir=tmp_path / "evidence",
    )
    client = TestClient(create_app(settings))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    presign = client.post(
        "/v1/uploads/presign",
        json={
            "filename": "zone.jpg",
            "content_type": "image/jpeg",
            "family": "fam",
            "zone_id": "zona_01",
        },
    )
    assert presign.status_code == 200
    upload = presign.json()

    put = client.put(upload["upload_url"], content=image_path.read_bytes(), headers={"content-type": "image/jpeg"})
    assert put.status_code == 200

    inspection = client.post(
        "/v1/inspections",
        json={
            "family": "fam",
            "zone_id": "zona_01",
            "image_uri": upload["object_uri"],
            "mold_id": "molde_1",
            "session_id": "sesion_1",
        },
    )
    assert inspection.status_code == 200
    body = inspection.json()
    assert body["status"] == "review"
    assert body["result"]["reason"] == "missing_best_model"
    assert body["result"]["mold_segmentation"]["ok"] is True
    assert body["identified_mold"] == "molde_1"
    assert body["identified_zone"] == "zona_01"
    assert len(body["mold_polygon"]) >= 3
    assert body["missing_regions"] == []


def test_cloud_api_crud_resource(tmp_path: Path):
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    created = client.post("/v1/families", json={"id": "molde_a", "name": "Molde A"})
    assert created.status_code == 200
    assert created.json()["id"] == "molde_a"

    fetched = client.get("/v1/families/molde_a")
    assert fetched.status_code == 200
    assert fetched.json()["data"]["name"] == "Molde A"


def test_firestore_nested_array_codec_preserves_missing_regions():
    payload = {
        "missing_regions": [[{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]],
        "findings": [{"region": [{"x": 0.1, "y": 0.2}]}],
    }

    encoded = _encode_firestore_nested_arrays(payload)

    assert encoded["missing_regions"] == [{"nested_array_items": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]}]
    assert _decode_firestore_nested_arrays(encoded) == payload


def test_cloud_api_mold_section_plan_persists_zone_views(tmp_path: Path):
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    response = client.post(
        "/v1/mold-section-plans/molde_a",
        json={
            "family": "fam",
            "mold_key": "molde_a",
            "name": "Molde A",
            "sections": [
                {
                    "id": "section_01_left",
                    "zone_id": "zona_01_left",
                    "label": "Zona 1 / izquierda",
                    "zone_index": 1,
                    "view": "left",
                },
                {
                    "id": "section_01_right",
                    "zone_id": "zona_01_right",
                    "label": "Zona 1 / derecha",
                    "zone_index": 1,
                    "view": "right",
                },
                {
                    "id": "section_02_front",
                    "zone_id": "zona_02_front",
                    "label": "Zona 2 / frente",
                    "zone_index": 2,
                    "view": "front",
                },
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["family"] == "fam"
    assert body["mold_key"] == "molde_a"
    assert body["section_count"] == 3
    assert body["required_count"] == 3
    assert [section["zone_id"] for section in body["sections"]] == ["zona_01_left", "zona_01_right", "zona_02_front"]

    fetched = client.get("/v1/mold-section-plans/molde_a?family=fam")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]

    listed = client.get("/v1/mold-section-plans?family=fam")
    assert listed.status_code == 200
    assert [record["id"] for record in listed.json()] == [body["id"]]

    zones = client.get("/v1/zones")
    assert zones.status_code == 200
    zone_lookup = {zone["id"]: zone for zone in zones.json()}
    assert zone_lookup["zona_01_left"]["family"] == "fam"
    assert zone_lookup["zona_01_left"]["view"] == "left"


def test_cloud_api_mold_validation_session_requires_all_zone_views(tmp_path: Path):
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    plan = client.post(
        "/v1/mold-section-plans/molde_a",
        json={
            "family": "fam",
            "mold_key": "molde_a",
            "sections": [
                {"id": "section_01_left", "zone_id": "zona_01_left", "label": "Zona 1 / izquierda", "zone_index": 1, "view": "left"},
                {"id": "section_01_right", "zone_id": "zona_01_right", "label": "Zona 1 / derecha", "zone_index": 1, "view": "right"},
            ],
        },
    )
    assert plan.status_code == 200

    created = client.post("/v1/mold-validation-sessions", json={"family": "fam", "mold_key": "molde_a"})
    assert created.status_code == 200
    session = created.json()
    assert session["status"] == "pending"
    assert session["required_count"] == 2
    assert set(session["missing_section_ids"]) == {"section_01_left", "section_01_right"}

    left = client.post(
        f"/v1/mold-validation-sessions/{session['id']}/sections/section_01_left",
        json={"zone_id": "zona_01_left", "status": "correct", "inspection_id": "insp_left", "image_uri": "local://left.jpg"},
    )
    assert left.status_code == 200
    body = left.json()
    assert body["status"] == "in_progress"
    assert body["completed_count"] == 1
    assert body["missing_section_ids"] == ["section_01_right"]

    right_retake = client.post(
        f"/v1/mold-validation-sessions/{session['id']}/sections/section_01_right",
        json={"zone_id": "zona_01_right", "status": "retake_photo", "inspection_id": "insp_right_bad"},
    )
    assert right_retake.status_code == 200
    assert right_retake.json()["status"] == "in_progress"
    assert right_retake.json()["missing_section_ids"] == ["section_01_right"]

    right_review = client.post(
        f"/v1/mold-validation-sessions/{session['id']}/sections/section_01_right",
        json={"zone_id": "zona_01_right", "status": "review", "inspection_id": "insp_right_review"},
    )
    assert right_review.status_code == 200
    complete = right_review.json()
    assert complete["status"] == "complete"
    assert complete["completed_count"] == 2
    assert complete["missing_section_ids"] == []
    assert complete["completed_at"]


def test_cloud_api_dataset_from_examples_generates_manifest_and_mask(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    ok_upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "dataset")
    fault_upload = _upload_file(client, image_paths["fault"], "fam", "zona_01", "dataset")
    response = client.post(
        "/v1/datasets/from-examples",
        json={
            "family": "fam",
            "zone_id": "zona_01",
            "name": "Dataset guiado",
            "ok_image_uris": [ok_upload["object_uri"]],
            "fault_image_uris": [fault_upload["object_uri"]],
            "mask": {"type": "auto"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_training"
    assert body["ok_count"] == 1
    assert body["fault_count"] == 1
    manifest_path = _local_upload_path(settings.local_state_dir, body["manifest_uri"])
    mask_path = _local_upload_path(settings.local_state_dir, body["mask_uri"])
    manifest_text = manifest_path.read_text()
    assert "label" in manifest_text
    assert ",ok," in manifest_text
    assert ",fault," in manifest_text
    rows = list(csv.DictReader(manifest_text.splitlines()))
    normalized_paths = [_local_upload_path(settings.local_state_dir, row["image_path"]) for row in rows]
    assert any("normalized_ok" in str(path) for path in normalized_paths)
    assert any("normalized_fault" in str(path) for path in normalized_paths)
    assert mask_path.exists()


def test_cloud_api_dataset_from_examples_requires_ok_and_fault(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    ok_upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "dataset")
    response = client.post(
        "/v1/datasets/from-examples",
        json={
            "family": "fam",
            "zone_id": "zona_01",
            "ok_image_uris": [ok_upload["object_uri"]],
            "fault_image_uris": [],
            "mask": {
                "type": "polygon",
                "points": [
                    {"x": 0.2, "y": 0.2},
                    {"x": 0.8, "y": 0.2},
                    {"x": 0.8, "y": 0.8},
                ],
            },
        },
    )

    assert response.status_code == 400
    assert "incorrect" in response.text.lower()


def test_cloud_api_capture_guidance_reports_alignment_instruction(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path, offset="left")
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "inspection")
    response = client.post(
        "/v1/capture-guidance",
        json={"family": "fam", "zone_id": "zona_01", "image_uri": upload["object_uri"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "Mueve a la derecha." in body["guidance"]
    polygon = body["alignment"]["mold_segmentation"]["polygon_normalized"]
    assert len(polygon) >= 3
    assert all(0 <= point["x"] <= 1 and 0 <= point["y"] <= 1 for point in polygon)


def test_cloud_api_zone_reference_expected_and_align_quality(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    upload = _upload_file(client, image_paths["ok"], "molde_demo", "frontal_zona_01", "reference")

    created = client.post(
        "/v1/zones/frontal_zona_01/reference",
        json={
            "family": "molde_demo",
            "image_uri": upload["object_uri"],
            "reference_id": "golden_sample",
            "tolerance": {"translation": 0.07},
        },
    )
    assert created.status_code == 200
    body = created.json()
    assert body["reference_id"] == "golden_sample"
    assert body["image_url"].endswith(f"/v1/uploads/{upload['upload_id']}/file")

    fetched = client.get("/v1/zones/frontal_zona_01/reference?family=molde_demo")
    assert fetched.status_code == 200
    assert fetched.json()["image_uri"] == upload["object_uri"]

    expected = client.get("/v1/zones/frontal_zona_01/expected?family=molde_demo")
    assert expected.status_code == 200
    assert {item["class_name"] for item in expected.json()} >= {"guide_post", "black_fastener"}
    generated_expected = client.get("/v1/zones/zona_01_front/expected?family=molde_demo")
    assert generated_expected.status_code == 200
    assert {item["class_name"] for item in generated_expected.json()} >= {"guide_post", "black_fastener"}

    align = client.post(
        "/v1/uploads/align-quality",
        json={"family": "molde_demo", "zone_id": "frontal_zona_01", "image_uri": upload["object_uri"], "reference_id": "golden_sample"},
    )
    assert align.status_code == 200
    assert align.json()["status"] in {"correct", "retake_photo"}
    assert "alignment" in align.json()


def test_cloud_api_inspection_uses_reference_piece_diff_without_model(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(
        local_state_dir=tmp_path / "state",
        model_registry_dir=tmp_path / "registry",
        evidence_dir=tmp_path / "evidence",
    )
    client = TestClient(create_app(settings))
    reference_upload = _upload_file(client, image_paths["ok"], "molde_demo", "zona_01_front", "reference")
    fault_upload = _upload_file(client, image_paths["fault"], "molde_demo", "zona_01_front", "inspection")

    reference = client.post(
        "/v1/zones/zona_01_front/reference",
        json={"family": "molde_demo", "image_uri": reference_upload["object_uri"], "reference_id": "golden_sample"},
    )
    assert reference.status_code == 200

    inspection = client.post(
        "/v1/inspections",
        json={
            "family": "molde_demo",
            "zone_id": "zona_01_front",
            "image_uri": fault_upload["object_uri"],
            "mold_id": "molde_demo",
        },
    )

    assert inspection.status_code == 200
    body = inspection.json()
    assert body["status"] == "review"
    assert body["result"]["reason"] == "reference_roi_diff_without_model"
    assert body["result"]["piece_inspection"]["missing_count"] > 0
    assert body["missing_regions"]
    assert all(_region_area(region) < 0.08 for region in body["missing_regions"])
    assert body["overlay_image_uri"]


def test_cloud_api_annotations_create_yolo_dataset_and_train(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "annotation")

    annotation = client.post(
        "/v1/annotations",
        json={
            "image_id": "img_001",
            "image_uri": upload["object_uri"],
            "family": "fam",
            "zone_id": "zona_01",
            "split": "train",
            "annotations": [
                {
                    "element_id": "bloque_ref_01",
                    "class_name": "block",
                    "bbox": [0.2, 0.25, 0.45, 0.55],
                    "status": "present",
                }
            ],
        },
    )
    assert annotation.status_code == 200
    assert annotation.json()["box_count"] == 1

    listed = client.get("/v1/annotations?family=fam&zone_id=zona_01")
    assert listed.status_code == 200
    assert listed.json()[0]["image_url"].endswith(f"/v1/uploads/{upload['upload_id']}/file")

    dataset = client.post(
        "/v1/segmenter-datasets/from-annotations",
        json={"family": "fam", "zone_id": "zona_01", "name": "YOLO desde anotaciones"},
    )
    assert dataset.status_code == 200
    dataset_body = dataset.json()
    assert dataset_body["status"] == "ready_for_training"
    assert dataset_body["box_count"] == 1
    root = Path(dataset_body["dataset_uri"].removeprefix("file://"))
    assert (root / "labels" / "train" / "000001.txt").read_text().startswith("0 0.325000 0.400000 0.250000 0.300000")

    training = client.post("/v1/inspector-training-jobs", json={"family": "fam", "zone_id": "zona_01", "dataset_id": dataset_body["id"]})
    assert training.status_code == 200
    assert training.json()["status"] == "queued"
    model_version = client.get("/v1/model_versions/best_fam_zona_01")
    assert model_version.status_code == 200
    model_body = model_version.json()
    assert model_body["model_uri"].endswith("/model.json")
    assert Path(model_body["model_uri"].removeprefix("file://")).exists()

    new_upload = _upload_file(client, image_paths["fault"], "fam", "zona_01", "annotation")
    draft = client.post(
        "/v1/annotations/auto-draft",
        json={"family": "fam", "zone_id": "zona_01", "image_uri": new_upload["object_uri"]},
    )
    assert draft.status_code == 200
    draft_body = draft.json()
    assert draft_body["source"] == "model"
    assert draft_body["model_version_id"] == "best_fam_zona_01"
    assert draft_body["annotations"][0]["class_name"] == "block"
    assert draft_body["annotations"][0]["bbox"] == [0.2, 0.25, 0.45, 0.55]
    assert draft_body["annotations"][0]["notes"].startswith("auto_draft_model:")


def test_cloud_api_auto_annotation_draft_reuses_latest_zone_annotation(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    template_upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "annotation")
    new_upload = _upload_file(client, image_paths["fault"], "fam", "zona_01", "annotation")

    saved = client.post(
        "/v1/annotations",
        json={
            "image_id": "template_img",
            "image_uri": template_upload["object_uri"],
            "family": "fam",
            "zone_id": "zona_01",
            "split": "train",
            "annotations": [
                {
                    "element_id": "guide_post",
                    "class_name": "guide_post",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "status": "present",
                }
            ],
        },
    )
    assert saved.status_code == 200

    draft = client.post(
        "/v1/annotations/auto-draft",
        json={"family": "fam", "zone_id": "zona_01", "image_uri": new_upload["object_uri"]},
    )
    assert draft.status_code == 200
    body = draft.json()
    assert body["source"] == "annotation_template"
    assert body["annotations"][0]["class_name"] == "guide_post"
    assert body["annotations"][0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert body["annotations"][0]["notes"].startswith("auto_draft_template:")


def test_cloud_api_segmenter_dataset_from_annotations(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "segmenter")
    response = client.post(
        "/v1/segmenter-datasets/from-annotations",
        json={
            "name": "Mold segmenter",
            "annotations": [
                {
                    "image_uri": upload["object_uri"],
                    "split": "train",
                    "polygon": [
                        {"x": 0.25, "y": 0.2},
                        {"x": 0.75, "y": 0.2},
                        {"x": 0.75, "y": 0.8},
                        {"x": 0.25, "y": 0.8},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_training"
    assert body["image_count"] == 1
    data_yaml = Path(body["data_yaml_uri"].removeprefix("file://"))
    assert data_yaml.exists()
    label = data_yaml.parent / "labels" / "train" / "000001.txt"
    assert label.read_text().startswith("0 0.250000 0.200000")


def test_cloud_api_segmenter_training_enqueues_async_job(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "segmenter")
    dataset = client.post(
        "/v1/segmenter-datasets/from-annotations",
        json={
            "name": "Mold segmenter",
            "annotations": [
                {
                    "image_uri": upload["object_uri"],
                    "split": "train",
                    "polygon": [
                        {"x": 0.25, "y": 0.2},
                        {"x": 0.75, "y": 0.2},
                        {"x": 0.75, "y": 0.8},
                        {"x": 0.25, "y": 0.8},
                    ],
                }
            ],
        },
    )
    assert dataset.status_code == 200
    response = client.post("/v1/segmenter-training-jobs", json={"dataset_id": dataset.json()["id"]})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["model_uri"].endswith("/best.pt")
    assert body["onnx_uri"].endswith("/best.onnx")
    assert body["training_command"][:3] == ["python3", "-m", "mold_inspection.cloud.segmenter_trainer"]


def test_cloud_api_recipe_and_inspector_training_candidates(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    ok_upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "dataset")
    fault_upload = _upload_file(client, image_paths["fault"], "fam", "zona_01", "dataset")
    dataset = client.post(
        "/v1/datasets/from-examples",
        json={
            "family": "fam",
            "zone_id": "zona_01",
            "ok_image_uris": [ok_upload["object_uri"]],
            "fault_image_uris": [fault_upload["object_uri"]],
            "mask": {"type": "auto"},
        },
    )
    assert dataset.status_code == 200

    recipe = client.post("/v1/recipes", json={"family": "fam", "zone_id": "zona_01", "name": "Receta Fam"})
    assert recipe.status_code == 200
    assert recipe.json()["status"] == "ready_for_training"

    training = client.post(
        "/v1/inspector-training-jobs",
        json={"family": "fam", "zone_id": "zona_01", "dataset_id": dataset.json()["id"]},
    )
    assert training.status_code == 200
    body = training.json()
    assert body["status"] == "queued"
    assert body["best_model_candidate_id"]
    assert len(body["candidates"]) == 3
    best = [candidate for candidate in body["candidates"] if candidate["promoted"]][0]
    assert best["metrics"]["false_pass_rate"] == 0.0
    model_version = client.get("/v1/model_versions/best_fam_zona_01")
    assert model_version.status_code == 200
    assert model_version.json()["candidate_id"] == best["id"]


def test_cloud_api_model_candidate_promote(tmp_path: Path):
    image_paths = _write_dataset_images(tmp_path)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    ok_upload = _upload_file(client, image_paths["ok"], "fam", "zona_01", "dataset")
    fault_upload = _upload_file(client, image_paths["fault"], "fam", "zona_01", "dataset")
    dataset = client.post(
        "/v1/datasets/from-examples",
        json={
            "family": "fam",
            "zone_id": "zona_01",
            "ok_image_uris": [ok_upload["object_uri"]],
            "fault_image_uris": [fault_upload["object_uri"]],
            "mask": {"type": "auto"},
        },
    ).json()
    training = client.post("/v1/inspector-training-jobs", json={"family": "fam", "zone_id": "zona_01", "dataset_id": dataset["id"]}).json()
    alternate = [candidate for candidate in training["candidates"] if not candidate["promoted"]][0]

    promoted = client.post(f"/v1/model-candidates/{alternate['id']}/promote", json={"notes": "manual override"})
    assert promoted.status_code == 200
    assert promoted.json()["promoted"] is True
    model_version = client.get("/v1/model_versions/best_fam_zona_01")
    assert model_version.json()["candidate_id"] == alternate["id"]


def test_cloud_api_public_dataset_import_registers_license_without_download(tmp_path: Path):
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))
    response = client.post("/v1/public-datasets/import", json={"dataset": "mvtec_ad", "category": "metal_nut"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "requires_download"
    assert body["license"] == "CC BY-NC-SA 4.0"
    assert body["intended_use"] == "benchmark_only"
    assert body["warnings"]


def test_cloud_api_public_dataset_import_maps_local_mvtec_layout(tmp_path: Path):
    root = tmp_path / "mvtec" / "metal_nut"
    ok_dir = root / "train" / "good"
    fault_dir = root / "test" / "missing"
    mask_dir = root / "ground_truth" / "missing"
    ok_dir.mkdir(parents=True)
    fault_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    _write_minimal_jpeg(ok_dir / "ok_1.jpg")
    _write_minimal_jpeg(fault_dir / "fault_1.jpg")
    _write_minimal_jpeg(mask_dir / "fault_1_mask.png")
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    response = client.post(
        "/v1/public-datasets/import",
        json={"dataset": "mvtec_ad", "category": "metal_nut", "local_root": str(tmp_path / "mvtec"), "max_items": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_benchmark"
    assert body["ok_count"] == 1
    assert body["fault_count"] == 1
    assert body["mask_count"] == 1
    assert body["mask_uri"]
    manifest_path = _local_upload_path(settings.local_state_dir, body["manifest_uri"])
    manifest_rows = list(csv.DictReader(manifest_path.read_text().splitlines()))
    assert {row["label"] for row in manifest_rows} == {"ok", "fault"}
    assert "mask_path" in manifest_rows[0]


def test_cloud_api_public_dataset_import_maps_local_visa_layout(tmp_path: Path):
    root = tmp_path / "visa" / "VisA_20220922"
    normal = root / "candle" / "Data" / "Images" / "Normal"
    anomaly = root / "candle" / "Data" / "Images" / "Anomaly"
    masks = root / "candle" / "Data" / "Masks" / "Anomaly"
    split = root / "split_csv"
    normal.mkdir(parents=True)
    anomaly.mkdir(parents=True)
    masks.mkdir(parents=True)
    split.mkdir(parents=True)
    _write_minimal_jpeg(normal / "n1.jpg")
    _write_minimal_jpeg(anomaly / "a1.jpg")
    _write_minimal_jpeg(masks / "a1.png")
    (split / "1cls.csv").write_text(
        "object,split,label,image,mask\n"
        "candle,train,normal,candle/Data/Images/Normal/n1.jpg,\n"
        "candle,test,anomaly,candle/Data/Images/Anomaly/a1.jpg,candle/Data/Masks/Anomaly/a1.png\n"
    )
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    response = client.post(
        "/v1/public-datasets/import",
        json={"dataset": "visa", "category": "candle", "local_root": str(tmp_path / "visa"), "max_items": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_benchmark"
    assert body["ok_count"] == 1
    assert body["fault_count"] == 1
    assert body["mask_count"] == 1
    assert body["mask_uri"]


def test_cloud_api_public_dataset_import_maps_local_kolektor_layout(tmp_path: Path):
    root = tmp_path / "KolektorSDD-boxes" / "kos01"
    root.mkdir(parents=True)
    _write_minimal_jpeg(root / "Part0.jpg")
    _write_minimal_jpeg(root / "Part1.jpg")
    try:
        import cv2
        import numpy as np
    except ImportError:
        (root / "Part0_label.bmp").write_bytes(b"BM" + bytes(52))
        (root / "Part1_label.bmp").write_bytes(b"BM" + bytes(52) + b"\xff")
    else:
        empty = np.zeros((32, 32), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        cv2.imwrite(str(root / "Part0_label.bmp"), empty)
        cv2.imwrite(str(root / "Part1_label.bmp"), mask)
    settings = CloudSettings(local_state_dir=tmp_path / "state", evidence_dir=tmp_path / "evidence")
    client = TestClient(create_app(settings))

    response = client.post(
        "/v1/public-datasets/import",
        json={"dataset": "kolektor_sdd", "category": "surface", "local_root": str(tmp_path / "KolektorSDD-boxes"), "max_items": 10},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_for_benchmark"
    assert body["ok_count"] == 1
    assert body["fault_count"] == 1
    assert body["mask_count"] == 2
    assert body["mask_uri"]


def _upload_file(client: TestClient, path: Path, family: str, zone_id: str, purpose: str) -> dict:
    presign = client.post(
        "/v1/uploads/presign",
        json={
            "filename": path.name,
            "content_type": "image/jpeg",
            "family": family,
            "zone_id": zone_id,
            "purpose": purpose,
        },
    )
    assert presign.status_code == 200
    upload = presign.json()
    put = client.put(upload["upload_url"], content=path.read_bytes(), headers={"content-type": "image/jpeg"})
    assert put.status_code == 200
    return upload


def _local_upload_path(state_dir: Path, object_uri: str) -> Path:
    upload_id = object_uri.removeprefix("local://")
    uploads = json.loads((state_dir / "metadata" / "uploads.json").read_text())
    return Path(uploads[upload_id]["path"])


def _region_area(region: list[dict[str, float]]) -> float:
    xs = [float(point["x"]) for point in region]
    ys = [float(point["y"]) for point in region]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _write_dataset_images(tmp_path: Path, offset: str = "center") -> dict[str, Path]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        _write_minimal_jpeg(tmp_path / "ok.jpg")
        _write_minimal_jpeg(tmp_path / "fault.jpg")
        return {"ok": tmp_path / "ok.jpg", "fault": tmp_path / "fault.jpg"}

    ok = np.full((720, 960, 3), 165, dtype=np.uint8)
    x1, x2 = (80, 520) if offset == "left" else (240, 720)
    cv2.rectangle(ok, (x1, 160), (x2, 560), (80, 80, 80), -1)
    cv2.circle(ok, (x1 + 160, 330), 65, (35, 35, 35), -1)
    fault = ok.copy()
    cv2.rectangle(fault, (x1 + 120, 270), (x1 + 220, 390), (165, 165, 165), -1)
    ok_path = tmp_path / "ok.jpg"
    fault_path = tmp_path / "fault.jpg"
    cv2.imwrite(str(ok_path), ok)
    cv2.imwrite(str(fault_path), fault)
    return {"ok": ok_path, "fault": fault_path}


def _write_minimal_jpeg(path: Path) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        path.write_bytes(
            bytes.fromhex(
                "ffd8ffe000104a46494600010101006000600000ffdb004300"
                "0302020302020303030304030304050805050404050a07070608"
                "0c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b10161011131415"
                "15150c0f171816141812141514ffdb0043010304040504050905"
                "0509140d0b0d1414141414141414141414141414141414141414"
                "1414141414141414141414141414141414141414141414141414"
                "141414141414ffc00011080001000103012200021101031101ff"
                "c4001400010000000000000000000000000000000000000000ff"
                "c4001410010000000000000000000000000000000000000000ff"
                "da000c03010002110311003f00b2c001ffd9"
            )
        )
        return
    image = np.full((720, 960, 3), 160, dtype=np.uint8)
    cv2.rectangle(image, (180, 140), (780, 580), (95, 95, 95), -1)
    cv2.circle(image, (430, 330), 90, (35, 35, 35), -1)
    cv2.imwrite(str(path), image)
