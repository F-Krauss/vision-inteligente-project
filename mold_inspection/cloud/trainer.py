from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import shutil
import tarfile

from mold_inspection.model_suite import train_model_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mold-cloud-trainer")
    parser.add_argument("--family", required=True)
    parser.add_argument("--zone-id", required=True)
    parser.add_argument("--manifest-uri", required=True)
    parser.add_argument("--mask-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--target", choices=["cloud-gpu"], default="cloud-gpu")
    parser.add_argument("--work-dir", default="/tmp/mold-cloud-training")
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = _prepare_manifest(args.manifest_uri, work_dir)
    mask = _materialize_uri(args.mask_uri, work_dir / "mask")
    registry_dir = work_dir / "registry"
    evidence_dir = work_dir / "evidence"

    report = train_model_suite(
        family=args.family,
        zone_id=args.zone_id,
        manifest_path=manifest,
        mask_path=mask,
        target=args.target,
        registry_dir=registry_dir,
        evidence_dir=evidence_dir,
    )

    best_dir = registry_dir / args.family / args.zone_id / "best_model"
    archive = work_dir / "best_model.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(best_dir, arcname="best_model")

    report_path = work_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _upload_file(archive, f"{args.output_uri.rstrip('/')}/best_model.tar.gz")
    _upload_file(report_path, f"{args.output_uri.rstrip('/')}/training_report.json")
    return 0


def _prepare_manifest(manifest_uri: str, work_dir: Path) -> Path:
    source = _materialize_uri(manifest_uri, work_dir / "manifest")
    output = work_dir / "manifest.local.csv"
    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    image_column = "image_path" if "image_path" in fieldnames else "path"
    if image_column not in fieldnames:
        raise ValueError("Manifest requires image_path or path column")
    for index, row in enumerate(rows):
        row[image_column] = str(_materialize_uri(row[image_column], image_dir / f"{index:06d}", base_dir=source.parent))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _materialize_uri(uri: str, destination_base: Path, base_dir: Path | None = None) -> Path:
    if uri.startswith("gs://"):
        client = _storage_client()
        bucket_name, blob_name = uri.removeprefix("gs://").split("/", 1)
        suffix = Path(blob_name).suffix
        destination = destination_base.with_suffix(suffix) if not destination_base.suffix else destination_base
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)
        return destination
    path = Path(uri.removeprefix("file://"))
    if not path.is_absolute() and base_dir:
        path = base_dir / path
    if not path.exists():
        raise ValueError(f"Missing training input: {uri}")
    return path


def _upload_file(path: Path, destination_uri: str) -> None:
    if destination_uri.startswith("gs://"):
        client = _storage_client()
        bucket_name, blob_name = destination_uri.removeprefix("gs://").split("/", 1)
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(path)
        return
    destination = Path(destination_uri.removeprefix("file://"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _storage_client():
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-storage is required for gs:// training URIs") from exc
    return storage.Client()


if __name__ == "__main__":
    raise SystemExit(main())
