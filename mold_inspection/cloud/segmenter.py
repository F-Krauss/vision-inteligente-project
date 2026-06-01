from __future__ import annotations

from pathlib import Path
import shutil

from .config import CloudSettings
from .schemas import (
    SegmenterDatasetCreateRequest,
    SegmenterDatasetRecord,
    SegmenterTrainingJobCreateRequest,
    SegmenterTrainingJobRecord,
)
from .storage import ObjectStorage
from .store import MetadataStore


def create_segmenter_dataset(
    request: SegmenterDatasetCreateRequest,
    settings: CloudSettings,
    objects: ObjectStorage,
    store: MetadataStore,
) -> SegmenterDatasetRecord:
    if not request.annotations:
        raise ValueError("El dataset del segmentador requiere al menos una imagen anotada.")
    for index, annotation in enumerate(request.annotations, start=1):
        if len(annotation.polygon) < 3:
            raise ValueError(f"La anotacion #{index} requiere al menos tres puntos.")

    record = SegmenterDatasetRecord(
        name=request.name,
        dataset_uri="",
        data_yaml_uri="",
        image_count=len(request.annotations),
        train_count=sum(1 for item in request.annotations if item.split == "train"),
        val_count=sum(1 for item in request.annotations if item.split == "val"),
        test_count=sum(1 for item in request.annotations if item.split == "test"),
    )
    root = settings.local_state_dir / "segmenter_datasets" / record.id
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index, annotation in enumerate(request.annotations, start=1):
        source = objects.materialize(annotation.image_uri)
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        image_name = f"{index:06d}{suffix}"
        image_path = root / "images" / annotation.split / image_name
        shutil.copy2(source, image_path)
        label_path = root / "labels" / annotation.split / f"{Path(image_name).stem}.txt"
        label_path.write_text(_yolo_segmentation_label(annotation.polygon) + "\n")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: mold",
                "",
            ]
        )
    )
    record.dataset_uri = f"file://{root}"
    record.data_yaml_uri = f"file://{data_yaml}"
    store.put("segmenter_datasets", record.id, record.model_dump())
    return record


def create_segmenter_training_job(
    request: SegmenterTrainingJobCreateRequest,
    settings: CloudSettings,
    store: MetadataStore,
) -> SegmenterTrainingJobRecord:
    dataset = store.get("segmenter_datasets", request.dataset_id)
    if not dataset:
        raise ValueError("Dataset de segmentador no encontrado.")
    data_yaml_uri = str(dataset.get("data_yaml_uri", ""))
    data_yaml = Path(data_yaml_uri.removeprefix("file://"))
    if not data_yaml.exists():
        raise ValueError("No existe data.yaml del dataset de segmentacion.")

    output_uri = request.output_uri or f"file://{settings.segmenter_model_path.parent}"
    model_uri = f"{output_uri.rstrip('/')}/best.pt"
    onnx_uri = f"{output_uri.rstrip('/')}/best.onnx"
    command = [
        "python3",
        "-m",
        "mold_inspection.cloud.segmenter_trainer",
        "--data-yaml",
        str(data_yaml),
        "--base-model",
        request.base_model,
        "--epochs",
        str(request.epochs),
        "--image-size",
        str(request.image_size),
        "--output-uri",
        output_uri,
    ]
    record = SegmenterTrainingJobRecord(
        dataset_id=request.dataset_id,
        status="queued",
        message="Trabajo de entrenamiento del segmentador registrado para ejecucion asincrona.",
        model_uri=model_uri,
        onnx_uri=onnx_uri,
        data_yaml_uri=data_yaml_uri,
        training_command=command,
        request=request.model_dump(),
    )
    store.put("segmenter_training_jobs", record.id, record.model_dump())
    return record


def _yolo_segmentation_label(points) -> str:
    values = ["0"]
    for point in points:
        values.append(f"{_clip(point.x):.6f}")
        values.append(f"{_clip(point.y):.6f}")
    return " ".join(values)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
