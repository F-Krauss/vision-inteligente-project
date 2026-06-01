from __future__ import annotations

from pathlib import Path
import json
import shutil

from .dataset import read_manifest
from .models import Box, InspectionConfig


def export_yolo_dataset(
    manifest_path: str | Path,
    annotations_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    config = InspectionConfig.load(config_path)
    class_names = config.class_names()
    class_to_id = {name: index for index, name in enumerate(class_names)}
    rows = read_manifest(manifest_path)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    counts = {"images": 0, "labels": 0}
    for row in rows:
        split = row.get("split") or "train"
        image_path = Path(row["image_path"])
        annotation_path = annotations_dir / f"{image_path.stem}.json"
        if not annotation_path.exists():
            continue

        image_out = output_dir / "images" / split / image_path.name
        label_out = output_dir / "labels" / split / f"{image_path.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_out)

        label_lines = _labels_from_annotation(annotation_path, class_to_id)
        label_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        counts["images"] += 1
        counts["labels"] += len(label_lines)

    data_yaml = output_dir / "data.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_block}\n"
    )
    (output_dir / "classes.json").write_text(json.dumps(class_to_id, indent=2) + "\n")
    return counts


def _labels_from_annotation(annotation_path: Path, class_to_id: dict[str, int]) -> list[str]:
    payload = json.loads(annotation_path.read_text())
    lines: list[str] = []

    for item in payload.get("annotations", []):
        if item.get("status", "present") != "present":
            continue
        class_name = item["class_name"]
        if class_name not in class_to_id:
            continue
        box = Box.from_list(item["bbox"])
        x_center = (box.x1 + box.x2) / 2
        y_center = (box.y1 + box.y2) / 2
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        lines.append(
            f"{class_to_id[class_name]} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return lines
