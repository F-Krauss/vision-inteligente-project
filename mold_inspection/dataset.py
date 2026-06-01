from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import random
import shutil
import subprocess
import time

from .models import ImageState

MANIFEST_FIELDS = [
    "image_path",
    "family",
    "mold_id",
    "session_id",
    "zone_id",
    "state",
    "source_path",
    "captured_at",
    "width",
    "height",
]


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: str | Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or _fields_for_rows(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_images(
    source: str | Path,
    manifest_path: str | Path,
    raw_dir: str | Path,
    family: str,
    mold_id: str,
    session_id: str,
    zone_id: str,
    state: str,
) -> list[dict[str, str]]:
    ImageState(state)
    source = Path(source)
    raw_dir = Path(raw_dir)
    manifest_path = Path(manifest_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = _source_files(source)
    rows = read_manifest(manifest_path)
    added: list[dict[str, str]] = []

    for index, image_path in enumerate(files, start=1):
        stem = f"{family}_{mold_id}_{session_id}_{zone_id}_{int(time.time())}_{index}"
        destination = raw_dir / family / mold_id / session_id / zone_id / f"{stem}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        final_path = _copy_as_jpeg_if_needed(image_path, destination)
        width, height = _image_size(final_path)
        row = {
            "image_path": str(final_path),
            "family": family,
            "mold_id": mold_id,
            "session_id": session_id,
            "zone_id": zone_id,
            "state": state,
            "source_path": str(image_path),
            "captured_at": str(int(time.time())),
            "width": str(width or ""),
            "height": str(height or ""),
        }
        rows.append(row)
        added.append(row)

    write_manifest(manifest_path, rows, MANIFEST_FIELDS)
    return added


def create_annotation_templates(
    manifest_path: str | Path,
    annotations_dir: str | Path,
    overwrite: bool = False,
) -> int:
    rows = read_manifest(manifest_path)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    created = 0

    for row in rows:
        image_path = Path(row["image_path"])
        annotation_path = annotations_dir / f"{image_path.stem}.json"
        if annotation_path.exists() and not overwrite:
            continue
        payload = {
            "image_path": row["image_path"],
            "family": row["family"],
            "mold_id": row["mold_id"],
            "session_id": row["session_id"],
            "zone_id": row["zone_id"],
            "state": row["state"],
            "annotations": [
                {
                    "element_id": "replace_with_expected_element_id",
                    "class_name": "replace_with_class_name",
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                    "status": "present",
                }
            ],
            "missing_expected": [],
        }
        annotation_path.write_text(json.dumps(payload, indent=2) + "\n")
        created += 1

    return created


def split_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> list[dict[str, str]]:
    rows = read_manifest(manifest_path)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["mold_id"], row["session_id"])].append(row)

    groups = list(grouped.items())
    random.Random(seed).shuffle(groups)

    total = len(groups)
    test_count = round(total * test_ratio)
    val_count = round(total * val_ratio)

    split_rows: list[dict[str, str]] = []
    for index, (_, group_rows) in enumerate(groups):
        if index < test_count:
            split = "test"
        elif index < test_count + val_count:
            split = "val"
        else:
            split = "train"
        for row in group_rows:
            split_row = dict(row)
            split_row["split"] = split
            split_rows.append(split_row)

    write_manifest(output_path, split_rows)
    return split_rows


def audit_split(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        groups[(row["family"], row["mold_id"], row["session_id"])].add(row.get("split", ""))

    leaks = {
        "/".join(group): sorted(splits)
        for group, splits in groups.items()
        if len(splits) > 1
    }
    return leaks


def _fields_for_rows(rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for field_name in MANIFEST_FIELDS + ["split"]:
        if any(field_name in row for row in rows):
            fields.append(field_name)
    for row in rows:
        for field_name in row:
            if field_name not in fields:
                fields.append(field_name)
    return fields


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in extensions)


def _copy_as_jpeg_if_needed(source: Path, destination: Path) -> Path:
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, destination)
        return destination

    try:
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(source), "--out", str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return destination
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install pillow or run on macOS with sips to convert non-JPEG images.") from exc

    with Image.open(source) as image:
        image.convert("RGB").save(destination, "JPEG", quality=95)
    return destination


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None

    with Image.open(path) as image:
        return image.size
