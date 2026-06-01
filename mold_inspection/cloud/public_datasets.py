from __future__ import annotations

from io import StringIO
from pathlib import Path
import csv

from .schemas import PublicDatasetImportRecord, PublicDatasetImportRequest
from .storage import ObjectStorage
from .store import MetadataStore


DATASET_CATALOG = {
    "mvtec_ad": {
        "source_url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "warning": "Uso no comercial; usar solo benchmark salvo permiso explicito.",
    },
    "visa": {
        "source_url": "https://dagshub.com/datasets/visual-anomaly-visa/",
        "license": "Research dataset; verify license before commercial use",
        "license_url": "https://dagshub.com/datasets/visual-anomaly-visa/",
        "warning": "Verificar licencia antes de usar fuera de benchmark.",
    },
    "kolektor_sdd": {
        "source_url": "https://www.vicos.si/resources/kolektorsdd/",
        "license": "Non-commercial research license",
        "license_url": "https://www.vicos.si/resources/kolektorsdd/",
        "warning": "Uso no comercial; usar solo benchmark salvo permiso explicito.",
    },
    "abo": {
        "source_url": "https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "warning": "Dataset de productos genericos; no representa moldes industriales.",
    },
}


def import_public_dataset(
    request: PublicDatasetImportRequest,
    objects: ObjectStorage,
    store: MetadataStore,
) -> PublicDatasetImportRecord:
    catalog = DATASET_CATALOG[request.dataset]
    family = request.family or f"benchmark_{request.dataset}"
    zone_id = request.zone_id or (request.category or "default")
    warnings = [catalog["warning"]]
    rows: list[dict[str, str]] = []
    mask_count = 0

    if request.local_root:
        rows, mask_count = _scan_local_dataset(Path(request.local_root), request, family, zone_id)
        status = "ready_for_benchmark" if rows else "empty_local_root"
        message = f"Importados {len(rows)} ejemplos desde carpeta local." if rows else "No se encontraron imagenes compatibles en local_root."
    else:
        status = "requires_download"
        message = "Dataset publico registrado. Descarga el dataset y vuelve a llamar con local_root para materializar manifest/masks."

    manifest_uri = None
    mask_uri = None
    if rows:
        manifest_uri = _write_manifest(objects, rows, family, zone_id)
        if any(row.get("mask_path") for row in rows):
            mask_uri = _write_mask_manifest(objects, rows, family, zone_id)

    record = PublicDatasetImportRecord(
        dataset=request.dataset,
        category=request.category,
        status=status,
        source_url=catalog["source_url"],
        license=catalog["license"],
        license_url=catalog["license_url"],
        family=family,
        zone_id=zone_id,
        manifest_uri=manifest_uri,
        mask_uri=mask_uri,
        ok_count=sum(1 for row in rows if row["label"] == "ok"),
        fault_count=sum(1 for row in rows if row["label"] == "fault"),
        mask_count=mask_count,
        message=message,
        warnings=warnings,
    )
    store.put("public_dataset_imports", record.id, record.model_dump())
    return record


def _scan_local_dataset(
    root: Path,
    request: PublicDatasetImportRequest,
    family: str,
    zone_id: str,
) -> tuple[list[dict[str, str]], int]:
    if not root.exists():
        raise ValueError(f"local_root does not exist: {root}")
    if request.dataset == "mvtec_ad":
        return _scan_mvtec(root, request, family, zone_id)
    if request.dataset == "visa":
        return _scan_visa(root, request, family, zone_id)
    if request.dataset == "kolektor_sdd":
        return _scan_kolektor(root, request, family, zone_id)
    return _scan_generic_anomaly(root, request, family, zone_id)


def _scan_mvtec(root: Path, request: PublicDatasetImportRequest, family: str, zone_id: str) -> tuple[list[dict[str, str]], int]:
    category_root = root / request.category if request.category and (root / request.category).exists() else root
    ok_paths = sorted((category_root / "train" / "good").glob("*"))
    rows = _rows(ok_paths, [], family, zone_id, request.max_items)
    fault_index = 0
    for defect_dir in sorted((category_root / "test").glob("*")):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        for image in sorted(defect_dir.glob("*"))[: request.max_items]:
            fault_index += 1
            row = _row(image, family, zone_id, "fault", fault_index)
            mask = category_root / "ground_truth" / defect_dir.name / f"{image.stem}_mask.png"
            if mask.exists():
                row["mask_path"] = str(mask)
            rows.append(row)
    mask_count = sum(1 for row in rows if row.get("mask_path"))
    return rows, mask_count


def _scan_generic_anomaly(root: Path, request: PublicDatasetImportRequest, family: str, zone_id: str) -> tuple[list[dict[str, str]], int]:
    ok_paths = _images_under(root / "ok") + _images_under(root / "good") + _images_under(root / "normal")
    fault_paths = _images_under(root / "fault") + _images_under(root / "defect") + _images_under(root / "anomaly")
    if not ok_paths or not fault_paths:
        all_images = _images_under(root)
        midpoint = max(1, len(all_images) // 2)
        ok_paths = all_images[:midpoint]
        fault_paths = all_images[midpoint:]
    mask_count = len(_images_under(root / "masks")) + len(_images_under(root / "ground_truth"))
    return _rows(ok_paths, fault_paths, family, zone_id, request.max_items), mask_count


def _scan_visa(root: Path, request: PublicDatasetImportRequest, family: str, zone_id: str) -> tuple[list[dict[str, str]], int]:
    dataset_root = root / "VisA_20220922" if (root / "VisA_20220922").exists() else root
    split_csv = dataset_root / "split_csv" / "1cls.csv"
    if split_csv.exists():
        rows: list[dict[str, str]] = []
        ok_count = 0
        fault_count = 0
        mask_count = 0
        with split_csv.open(newline="") as handle:
            for source in csv.DictReader(handle):
                if request.category and source.get("object") != request.category:
                    continue
                label = "ok" if source.get("label") == "normal" else "fault"
                if label == "ok" and ok_count >= request.max_items:
                    continue
                if label == "fault" and fault_count >= request.max_items:
                    continue
                image = dataset_root / str(source.get("image", ""))
                if not image.exists():
                    continue
                mask_path = dataset_root / str(source.get("mask") or "")
                if source.get("mask") and mask_path.exists():
                    mask_count += 1
                if label == "ok":
                    ok_count += 1
                    index = ok_count
                else:
                    fault_count += 1
                    index = fault_count
                row = _row(image, family, zone_id, label, index)
                if source.get("mask") and mask_path.exists():
                    row["mask_path"] = str(mask_path)
                rows.append(row)
        return rows, mask_count

    category_root = dataset_root / request.category if request.category and (dataset_root / request.category).exists() else dataset_root
    ok_paths = _images_under(category_root / "Data" / "Images" / "Normal")
    fault_paths = _images_under(category_root / "Data" / "Images" / "Anomaly")
    mask_paths = _images_under(category_root / "Data" / "Masks" / "Anomaly")
    mask_by_stem = {path.stem: path for path in mask_paths}
    rows = _rows(ok_paths, [], family, zone_id, request.max_items)
    for index, path in enumerate(fault_paths[: request.max_items], start=1):
        row = _row(path, family, zone_id, "fault", index)
        mask = mask_by_stem.get(path.stem)
        if mask:
            row["mask_path"] = str(mask)
        rows.append(row)
    return rows, len(mask_paths)


def _scan_kolektor(root: Path, request: PublicDatasetImportRequest, family: str, zone_id: str) -> tuple[list[dict[str, str]], int]:
    image_paths = sorted(path for path in root.rglob("*.jpg") if "_label" not in path.stem)
    ok_paths: list[Path] = []
    fault_paths: list[Path] = []
    mask_count = 0
    for image in image_paths:
        label = image.with_name(f"{image.stem}_label.bmp")
        has_defect = _mask_has_foreground(label)
        if label.exists():
            mask_count += 1
        if has_defect:
            fault_paths.append(image)
        else:
            ok_paths.append(image)
    rows = _rows(ok_paths, [], family, zone_id, request.max_items)
    for index, image in enumerate(fault_paths[: request.max_items], start=1):
        row = _row(image, family, zone_id, "fault", index)
        label = image.with_name(f"{image.stem}_label.bmp")
        if label.exists():
            row["mask_path"] = str(label)
        rows.append(row)
    return rows, mask_count


def _images_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)


def _rows(ok_paths: list[Path], fault_paths: list[Path], family: str, zone_id: str, max_items: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, paths in [("ok", ok_paths), ("fault", fault_paths)]:
        for index, path in enumerate(paths[:max_items], start=1):
            rows.append(_row(path, family, zone_id, label, index))
    return rows


def _row(path: Path, family: str, zone_id: str, label: str, index: int) -> dict[str, str]:
    return {
        "image_path": str(path),
        "family": family,
        "zone_id": zone_id,
        "label": label,
        "mold_id": f"{family}_{label}_{index}",
        "session_id": f"public_dataset_{zone_id}_{label}_{index}",
        "split": "train" if index % 5 else "val",
    }


def _mask_has_foreground(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        from PIL import Image
    except ImportError:
        return any(byte not in {0, 10, 13, 26, 32, 66, 77} for byte in path.read_bytes()[54:])
    with Image.open(path) as image:
        extrema = image.convert("L").getextrema()
    return bool(extrema and extrema[1] > 0)


def _write_manifest(objects: ObjectStorage, rows: list[dict[str, str]], family: str, zone_id: str) -> str:
    fields = ["image_path", "family", "zone_id", "label", "mold_id", "session_id", "split"]
    if any(row.get("mask_path") for row in rows):
        fields.append("mask_path")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    upload = objects.create_upload("public_manifest.csv", "text/csv", family, zone_id, "dataset")
    return objects.write_upload(upload.upload_id, buffer.getvalue().encode("utf-8"), "text/csv")


def _write_mask_manifest(objects: ObjectStorage, rows: list[dict[str, str]], family: str, zone_id: str) -> str:
    fields = ["image_path", "mask_path", "label"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        [
            {"image_path": row["image_path"], "mask_path": row["mask_path"], "label": row["label"]}
            for row in rows
            if row.get("mask_path")
        ]
    )
    upload = objects.create_upload("public_masks.csv", "text/csv", family, zone_id, "dataset")
    return objects.write_upload(upload.upload_id, buffer.getvalue().encode("utf-8"), "text/csv")
