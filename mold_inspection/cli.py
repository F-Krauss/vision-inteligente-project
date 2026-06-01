from __future__ import annotations

from pathlib import Path
import argparse
import json

from .anomaly_reference import (
    inspect_anomaly_images,
    set_zone_mask,
    train_anomaly_model,
    write_anomaly_report,
)
from .dataset import (
    add_images,
    audit_split,
    create_annotation_templates,
    read_manifest,
    split_manifest,
)
from .golden_reference import create_reference, inspect_against_reference, write_golden_report
from .model_suite import export_best_model, inspect_best_model, train_model_suite
from .yolo_export import export_yolo_dataset
from .yolo_runtime import inspect_images, train_yolo, write_report

DEFAULT_CONFIG = Path("config/inspection.json")
DEFAULT_MANIFEST = Path("data/manifest.csv")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_ANNOTATIONS = Path("data/annotations")
DEFAULT_SPLIT_MANIFEST = Path("data/splits/manifest.csv")
DEFAULT_YOLO_DIR = Path("data/yolo")
DEFAULT_REFERENCES_DIR = Path("data/references")
DEFAULT_EVIDENCE_DIR = Path("reports/evidence")
DEFAULT_ANOMALY_DIR = Path("data/anomaly")
DEFAULT_ANOMALY_EVIDENCE_DIR = Path("reports/anomaly_evidence")
DEFAULT_MODEL_REGISTRY_DIR = Path("data/model_registry")
DEFAULT_BEST_EVIDENCE_DIR = Path("reports/best_model_evidence")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mold-inspect")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")

    add_parser = subparsers.add_parser("add-images")
    add_parser.add_argument("--source", required=True)
    add_parser.add_argument("--family", required=True)
    add_parser.add_argument("--mold-id", required=True)
    add_parser.add_argument("--session-id", required=True)
    add_parser.add_argument("--zone-id", required=True)
    add_parser.add_argument("--state", choices=["correct", "incorrect", "simulated_fault"], required=True)
    add_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    add_parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))

    template_parser = subparsers.add_parser("label-template")
    template_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    template_parser.add_argument("--annotations-dir", default=str(DEFAULT_ANNOTATIONS))
    template_parser.add_argument("--overwrite", action="store_true")

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    split_parser.add_argument("--out", default=str(DEFAULT_SPLIT_MANIFEST))
    split_parser.add_argument("--val-ratio", type=float, default=0.15)
    split_parser.add_argument("--test-ratio", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=13)

    audit_parser = subparsers.add_parser("audit-split")
    audit_parser.add_argument("--manifest", default=str(DEFAULT_SPLIT_MANIFEST))

    export_parser = subparsers.add_parser("export-yolo")
    export_parser.add_argument("--manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    export_parser.add_argument("--annotations-dir", default=str(DEFAULT_ANNOTATIONS))
    export_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    export_parser.add_argument("--out", default=str(DEFAULT_YOLO_DIR))

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-yaml", default=str(DEFAULT_YOLO_DIR / "data.yaml"))
    train_parser.add_argument("--weights", default="yolo11n.pt")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--image-size", type=int, default=960)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--weights", required=True)
    inspect_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    inspect_parser.add_argument("--family", required=True)
    inspect_parser.add_argument("--zone-id", required=True)
    inspect_parser.add_argument("--images", nargs="+", required=True)
    inspect_parser.add_argument("--confidence", type=float, default=0.25)
    inspect_parser.add_argument("--out", default="reports/inspection.json")

    reference_parser = subparsers.add_parser("set-reference")
    reference_parser.add_argument("--image", required=True)
    reference_parser.add_argument("--family", required=True)
    reference_parser.add_argument("--zone-id", required=True)
    reference_parser.add_argument("--reference-id", default="default")
    reference_parser.add_argument("--references-dir", default=str(DEFAULT_REFERENCES_DIR))

    references_parser = subparsers.add_parser("set-references")
    references_parser.add_argument("--images", nargs="+", required=True)
    references_parser.add_argument("--family", required=True)
    references_parser.add_argument("--zone-id", required=True)
    references_parser.add_argument("--reference-prefix", default="ref")
    references_parser.add_argument("--references-dir", default=str(DEFAULT_REFERENCES_DIR))

    golden_parser = subparsers.add_parser("inspect-golden")
    golden_parser.add_argument("--family", required=True)
    golden_parser.add_argument("--zone-id", required=True)
    golden_parser.add_argument("--images", nargs="+", required=True)
    golden_parser.add_argument("--references-dir", default=str(DEFAULT_REFERENCES_DIR))
    golden_parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    golden_parser.add_argument("--min-similarity", type=float, default=0.90)
    golden_parser.add_argument("--min-keypoints", type=int, default=40)
    golden_parser.add_argument("--min-inlier-ratio", type=float, default=0.18)
    golden_parser.add_argument("--max-brightness-delta", type=float, default=65.0)
    golden_parser.add_argument("--min-blur-score", type=float, default=35.0)
    golden_parser.add_argument("--difference-threshold", type=int, default=38)
    golden_parser.add_argument("--min-region-area", type=int, default=350)
    golden_parser.add_argument("--min-comparable-similarity", type=float, default=0.70)
    golden_parser.add_argument("--max-difference-area-ratio", type=float, default=0.60)
    golden_parser.add_argument("--out", default="reports/golden_inspection.json")

    mask_parser = subparsers.add_parser("set-zone-mask")
    mask_parser.add_argument("--family", required=True)
    mask_parser.add_argument("--zone-id", required=True)
    mask_parser.add_argument("--mask", required=True)
    mask_parser.add_argument("--anomaly-dir", default=str(DEFAULT_ANOMALY_DIR))

    train_anomaly_parser = subparsers.add_parser("train-anomaly")
    train_anomaly_parser.add_argument("--family", required=True)
    train_anomaly_parser.add_argument("--zone-id", required=True)
    train_anomaly_parser.add_argument("--images", nargs="+", required=True)
    train_anomaly_parser.add_argument("--mask")
    train_anomaly_parser.add_argument("--anomaly-dir", default=str(DEFAULT_ANOMALY_DIR))
    train_anomaly_parser.add_argument("--feature-backend", choices=["resnet18", "classical"], default="resnet18")
    train_anomaly_parser.add_argument("--image-size", type=int, default=512)
    train_anomaly_parser.add_argument("--max-bank-patches", type=int, default=20000)
    train_anomaly_parser.add_argument("--location-weight", type=float, default=0.15)
    train_anomaly_parser.add_argument("--anomaly-threshold", type=float)
    train_anomaly_parser.add_argument("--heatmap-threshold", type=float)
    train_anomaly_parser.add_argument("--min-keypoints", type=int, default=40)
    train_anomaly_parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    train_anomaly_parser.add_argument("--max-brightness-delta", type=float, default=70.0)
    train_anomaly_parser.add_argument("--min-blur-score", type=float, default=25.0)
    train_anomaly_parser.add_argument("--min-mask-coverage", type=float, default=0.55)
    train_anomaly_parser.add_argument("--min-region-area", type=int, default=350)

    inspect_anomaly_parser = subparsers.add_parser("inspect-anomaly")
    inspect_anomaly_parser.add_argument("--family", required=True)
    inspect_anomaly_parser.add_argument("--zone-id", required=True)
    inspect_anomaly_parser.add_argument("--images", nargs="+", required=True)
    inspect_anomaly_parser.add_argument("--anomaly-dir", default=str(DEFAULT_ANOMALY_DIR))
    inspect_anomaly_parser.add_argument("--evidence-dir", default=str(DEFAULT_ANOMALY_EVIDENCE_DIR))
    inspect_anomaly_parser.add_argument("--anomaly-threshold", type=float)
    inspect_anomaly_parser.add_argument("--heatmap-threshold", type=float)
    inspect_anomaly_parser.add_argument("--min-region-area", type=int)
    inspect_anomaly_parser.add_argument("--out", default="reports/anomaly_inspection.json")

    suite_parser = subparsers.add_parser("train-model-suite")
    suite_parser.add_argument("--family", required=True)
    suite_parser.add_argument("--zone-id", required=True)
    suite_parser.add_argument("--manifest", required=True)
    suite_parser.add_argument("--mask", required=True)
    suite_parser.add_argument("--target", choices=["raspberry-pi", "cloud-gpu"], default="raspberry-pi")
    suite_parser.add_argument("--registry-dir", default=str(DEFAULT_MODEL_REGISTRY_DIR))
    suite_parser.add_argument("--evidence-dir", default="reports/model_suite_evidence")
    suite_parser.add_argument("--max-false-accept-rate", type=float, default=0.0)

    export_best_parser = subparsers.add_parser("export-best")
    export_best_parser.add_argument("--family", required=True)
    export_best_parser.add_argument("--zone-id", required=True)
    export_best_parser.add_argument("--target", choices=["raspberry-pi", "cloud-gpu"], default="raspberry-pi")
    export_best_parser.add_argument("--registry-dir", default=str(DEFAULT_MODEL_REGISTRY_DIR))

    inspect_best_parser = subparsers.add_parser("inspect-best")
    inspect_best_parser.add_argument("--family", required=True)
    inspect_best_parser.add_argument("--zone-id", required=True)
    inspect_best_parser.add_argument("--images", nargs="+", required=True)
    inspect_best_parser.add_argument("--registry-dir", default=str(DEFAULT_MODEL_REGISTRY_DIR))
    inspect_best_parser.add_argument("--evidence-dir", default=str(DEFAULT_BEST_EVIDENCE_DIR))
    inspect_best_parser.add_argument("--out", default="reports/best_inspection.json")

    args = parser.parse_args(argv)

    if args.command == "init":
        _init_project()
    elif args.command == "add-images":
        added = add_images(
            source=args.source,
            manifest_path=args.manifest,
            raw_dir=args.raw_dir,
            family=args.family,
            mold_id=args.mold_id,
            session_id=args.session_id,
            zone_id=args.zone_id,
            state=args.state,
        )
        print(json.dumps({"added": len(added)}, indent=2))
    elif args.command == "label-template":
        created = create_annotation_templates(args.manifest, args.annotations_dir, args.overwrite)
        print(json.dumps({"created": created}, indent=2))
    elif args.command == "split":
        rows = split_manifest(args.manifest, args.out, args.val_ratio, args.test_ratio, args.seed)
        print(json.dumps({"rows": len(rows), "out": args.out}, indent=2))
    elif args.command == "audit-split":
        leaks = audit_split(read_manifest(args.manifest))
        print(json.dumps({"leaks": leaks, "ok": not leaks}, indent=2))
        return 1 if leaks else 0
    elif args.command == "export-yolo":
        counts = export_yolo_dataset(args.manifest, args.annotations_dir, args.config, args.out)
        print(json.dumps(counts, indent=2))
    elif args.command == "train":
        train_yolo(args.data_yaml, args.weights, args.epochs, args.image_size)
    elif args.command == "inspect":
        reports = inspect_images(
            weights=args.weights,
            config_path=args.config,
            family=args.family,
            zone_id=args.zone_id,
            images=args.images,
            confidence=args.confidence,
        )
        write_report(args.out, reports)
        print(json.dumps({"reports": len(reports), "out": args.out}, indent=2))
    elif args.command == "set-reference":
        reference_path = create_reference(
            source_image=args.image,
            family=args.family,
            zone_id=args.zone_id,
            reference_id=args.reference_id,
            references_dir=args.references_dir,
        )
        print(json.dumps({"reference": str(reference_path)}, indent=2))
    elif args.command == "set-references":
        reference_paths = []
        for index, image in enumerate(args.images, start=1):
            reference_path = create_reference(
                source_image=image,
                family=args.family,
                zone_id=args.zone_id,
                reference_id=f"{args.reference_prefix}_{index:03d}",
                references_dir=args.references_dir,
            )
            reference_paths.append(str(reference_path))
        print(json.dumps({"references": reference_paths, "count": len(reference_paths)}, indent=2))
    elif args.command == "inspect-golden":
        reports = []
        retake_required = False
        for image in args.images:
            result = inspect_against_reference(
                image_path=image,
                family=args.family,
                zone_id=args.zone_id,
                references_dir=args.references_dir,
                evidence_dir=args.evidence_dir,
                min_similarity=args.min_similarity,
                min_keypoints=args.min_keypoints,
                min_inlier_ratio=args.min_inlier_ratio,
                max_brightness_delta=args.max_brightness_delta,
                min_blur_score=args.min_blur_score,
                difference_threshold=args.difference_threshold,
                min_region_area=args.min_region_area,
                min_comparable_similarity=args.min_comparable_similarity,
                max_difference_area_ratio=args.max_difference_area_ratio,
            )
            if result.status == "retake_photo":
                retake_required = True
            reports.append(
                {
                    "image_path": image,
                    "family": args.family,
                    "zone_id": args.zone_id,
                    "result": result.as_dict(),
                }
            )
        write_golden_report(args.out, reports)
        print(json.dumps({"reports": len(reports), "out": args.out, "retake_required": retake_required}, indent=2))
        return 2 if retake_required else 0
    elif args.command == "set-zone-mask":
        mask_path = set_zone_mask(args.family, args.zone_id, args.mask, args.anomaly_dir)
        print(json.dumps({"mask": str(mask_path)}, indent=2))
    elif args.command == "train-anomaly":
        profile = train_anomaly_model(
            family=args.family,
            zone_id=args.zone_id,
            images=args.images,
            mask_path=args.mask,
            anomaly_dir=args.anomaly_dir,
            feature_backend=args.feature_backend,
            image_size=args.image_size,
            max_bank_patches=args.max_bank_patches,
            location_weight=args.location_weight,
            anomaly_threshold=args.anomaly_threshold,
            heatmap_threshold=args.heatmap_threshold,
            min_keypoints=args.min_keypoints,
            min_inlier_ratio=args.min_inlier_ratio,
            max_brightness_delta=args.max_brightness_delta,
            min_blur_score=args.min_blur_score,
            min_mask_coverage=args.min_mask_coverage,
            min_region_area=args.min_region_area,
        )
        print(json.dumps({"profile": profile}, indent=2))
    elif args.command == "inspect-anomaly":
        reports = inspect_anomaly_images(
            family=args.family,
            zone_id=args.zone_id,
            images=args.images,
            anomaly_dir=args.anomaly_dir,
            evidence_dir=args.evidence_dir,
            anomaly_threshold=args.anomaly_threshold,
            heatmap_threshold=args.heatmap_threshold,
            min_region_area=args.min_region_area,
        )
        retake_required = any(report["result"]["status"] == "retake_photo" for report in reports)
        write_anomaly_report(args.out, reports)
        print(json.dumps({"reports": len(reports), "out": args.out, "retake_required": retake_required}, indent=2))
        return 2 if retake_required else 0
    elif args.command == "train-model-suite":
        report = train_model_suite(
            family=args.family,
            zone_id=args.zone_id,
            manifest_path=args.manifest,
            mask_path=args.mask,
            target=args.target,
            registry_dir=args.registry_dir,
            evidence_dir=args.evidence_dir,
            max_false_accept_rate=args.max_false_accept_rate,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "export-best":
        exported = export_best_model(
            family=args.family,
            zone_id=args.zone_id,
            target=args.target,
            registry_dir=args.registry_dir,
        )
        print(json.dumps(exported, indent=2))
    elif args.command == "inspect-best":
        reports = inspect_best_model(
            family=args.family,
            zone_id=args.zone_id,
            images=args.images,
            registry_dir=args.registry_dir,
            evidence_dir=args.evidence_dir,
            out=args.out,
        )
        retake_required = any(report["result"]["status"] == "retake_photo" for report in reports)
        print(json.dumps({"reports": len(reports), "out": args.out, "retake_required": retake_required}, indent=2))
        return 2 if retake_required else 0

    return 0


def _init_project() -> None:
    for path in [
        Path("config"),
        DEFAULT_RAW_DIR,
        DEFAULT_ANNOTATIONS,
        Path("data/splits"),
        DEFAULT_YOLO_DIR,
        DEFAULT_REFERENCES_DIR,
        DEFAULT_ANOMALY_DIR,
        DEFAULT_MODEL_REGISTRY_DIR,
        Path("models"),
        Path("reports"),
        DEFAULT_EVIDENCE_DIR,
        DEFAULT_ANOMALY_EVIDENCE_DIR,
        DEFAULT_BEST_EVIDENCE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_CONFIG.exists():
        DEFAULT_CONFIG.write_text(json.dumps(_sample_config(), indent=2) + "\n")

    dataset_doc = Path("data/README_DATASET.md")
    if not dataset_doc.exists():
        dataset_doc.write_text(
            "# Dataset\n\n"
            "Capture cada molde por zonas con una secuencia fija. No mezcle sesiones del "
            "mismo molde entre train, val y test.\n"
        )

    print(json.dumps({"created": True, "config": str(DEFAULT_CONFIG)}, indent=2))


def _sample_config() -> dict:
    return {
        "families": {
            "familia_demo": {
                "zones": {
                    "zona_01": {
                        "description": "Zona inicial de ejemplo. Ajustar antes de entrenar.",
                        "expected": [
                            {
                                "id": "bloque_ref_01",
                                "class_name": "block",
                                "roi": [0.10, 0.10, 0.35, 0.35],
                                "min_confidence": 0.55,
                                "min_overlap": 0.25,
                                "critical": True,
                            },
                            {
                                "id": "tornillo_ref_01",
                                "class_name": "screw",
                                "roi": [0.50, 0.40, 0.58, 0.48],
                                "min_confidence": 0.50,
                                "min_overlap": 0.20,
                                "critical": True,
                            },
                        ],
                    }
                }
            }
        }
    }


if __name__ == "__main__":
    raise SystemExit(main())
