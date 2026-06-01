from __future__ import annotations

from pathlib import Path
import argparse
import shutil


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mold-segmenter-trainer")
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--base-model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--work-dir", default="/tmp/mold-segmenter-training")
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required to train the mold segmenter") from exc

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.base_model)
    result = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.image_size,
        project=str(work_dir),
        name="run",
        exist_ok=True,
    )
    run_dir = Path(getattr(result, "save_dir", work_dir / "run"))
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"Training did not produce best.pt: {best_pt}")

    trained = YOLO(str(best_pt))
    trained.export(format="onnx", imgsz=args.image_size)
    best_onnx = best_pt.with_suffix(".onnx")
    if not best_onnx.exists():
        best_onnx = run_dir / "weights" / "best.onnx"

    _copy_to_uri(best_pt, f"{args.output_uri.rstrip('/')}/best.pt")
    if best_onnx.exists():
        _copy_to_uri(best_onnx, f"{args.output_uri.rstrip('/')}/best.onnx")
    return 0


def _copy_to_uri(source: Path, destination_uri: str) -> None:
    destination = Path(destination_uri.removeprefix("file://"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


if __name__ == "__main__":
    raise SystemExit(main())
