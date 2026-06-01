from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import shutil
import time

from .anomaly_reference import inspect_anomaly_model_dir, train_anomaly_model, write_anomaly_report


MODEL_SUITE_VERSION = 1


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    feature_backend: str
    image_size: int
    max_bank_patches: int
    location_weight: float
    max_ram_mb: int
    target_latency_s: float


RASPBERRY_CANDIDATES = [
    CandidateSpec(
        id="patchcore_classical_fast",
        feature_backend="classical",
        image_size=320,
        max_bank_patches=5000,
        location_weight=0.10,
        max_ram_mb=512,
        target_latency_s=4.0,
    ),
    CandidateSpec(
        id="patchcore_classical_balanced",
        feature_backend="classical",
        image_size=384,
        max_bank_patches=10000,
        location_weight=0.15,
        max_ram_mb=1024,
        target_latency_s=6.0,
    ),
    CandidateSpec(
        id="patchcore_classical_detail",
        feature_backend="classical",
        image_size=512,
        max_bank_patches=15000,
        location_weight=0.18,
        max_ram_mb=1536,
        target_latency_s=8.0,
    ),
]

CLOUD_GPU_CANDIDATES = [
    CandidateSpec(
        id="patchcore_resnet18_balanced",
        feature_backend="resnet18",
        image_size=512,
        max_bank_patches=30000,
        location_weight=0.15,
        max_ram_mb=4096,
        target_latency_s=5.0,
    ),
    CandidateSpec(
        id="patchcore_resnet18_detail",
        feature_backend="resnet18",
        image_size=768,
        max_bank_patches=50000,
        location_weight=0.18,
        max_ram_mb=8192,
        target_latency_s=5.0,
    ),
    CandidateSpec(
        id="patchcore_classical_guardrail",
        feature_backend="classical",
        image_size=512,
        max_bank_patches=20000,
        location_weight=0.15,
        max_ram_mb=1536,
        target_latency_s=3.0,
    ),
]


def train_model_suite(
    family: str,
    zone_id: str,
    manifest_path: str | Path,
    mask_path: str | Path,
    target: str = "raspberry-pi",
    registry_dir: str | Path = "data/model_registry",
    evidence_dir: str | Path = "reports/model_suite_evidence",
    max_false_accept_rate: float = 0.0,
) -> dict[str, Any]:
    candidates = _candidate_specs(target)

    rows = _load_manifest(manifest_path, family, zone_id)
    split_rows = _ensure_splits(rows)
    train_ok = [row["image_path"] for row in split_rows if row["split"] == "train" and row["label"] == "ok"]
    eval_rows = [row for row in split_rows if row["split"] in {"val", "test"}]
    if not train_ok:
        raise ValueError("Manifest must include at least one training image labeled ok")
    if not eval_rows:
        eval_rows = [row for row in split_rows if row["split"] == "train"]

    suite_dir = _suite_dir(registry_dir, family, zone_id)
    candidates_dir = suite_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: list[dict[str, Any]] = []
    for spec in candidates:
        candidate_dir = candidates_dir / spec.id
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        start_train = time.perf_counter()
        profile = train_anomaly_model(
            family=family,
            zone_id=zone_id,
            images=train_ok,
            mask_path=mask_path,
            anomaly_dir=candidate_dir.parent,
            feature_backend=spec.feature_backend,
            image_size=spec.image_size,
            max_bank_patches=spec.max_bank_patches,
            location_weight=spec.location_weight,
        )
        generated_dir = candidate_dir.parent / family / zone_id
        if generated_dir != candidate_dir:
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
            shutil.move(str(generated_dir), str(candidate_dir))
            _cleanup_empty_parents(generated_dir.parent, stop_at=candidate_dir.parent)

        train_seconds = time.perf_counter() - start_train
        metrics = _evaluate_candidate(
            candidate_dir=candidate_dir,
            spec=spec,
            family=family,
            zone_id=zone_id,
            rows=eval_rows,
            evidence_dir=Path(evidence_dir) / family / zone_id / spec.id,
            max_false_accept_rate=max_false_accept_rate,
        )
        candidate_report = {
            "candidate_id": spec.id,
            "target": target,
            "profile": profile,
            "train_seconds": round(train_seconds, 4),
            "metrics": metrics,
        }
        (candidate_dir / "evaluation.json").write_text(json.dumps(candidate_report, indent=2) + "\n")
        leaderboard.append(candidate_report)

    leaderboard.sort(key=_candidate_sort_key)
    selected = leaderboard[0]
    suite_report = {
        "version": MODEL_SUITE_VERSION,
        "family": family,
        "zone_id": zone_id,
        "target": target,
        "selected_candidate": selected["candidate_id"],
        "production_ready": bool(selected["metrics"]["production_ready"]),
        "selection_policy": [
            "minimize false_accept_rate",
            "maximize fault_recall",
            "minimize false_reject_rate",
            "minimize avg_latency_s",
            "minimize estimated_ram_mb",
        ],
        "leaderboard": leaderboard,
    }
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2) + "\n")
    (suite_dir / "evaluation_report.json").write_text(json.dumps(suite_report, indent=2) + "\n")
    export_best_model(family, zone_id, target=target, registry_dir=registry_dir)
    return suite_report


def export_best_model(
    family: str,
    zone_id: str,
    target: str = "raspberry-pi",
    registry_dir: str | Path = "data/model_registry",
) -> dict[str, Any]:
    _candidate_specs(target)

    suite_dir = _suite_dir(registry_dir, family, zone_id)
    report = json.loads((suite_dir / "evaluation_report.json").read_text())
    selected_id = report["selected_candidate"]
    source_dir = suite_dir / "candidates" / selected_id
    best_dir = suite_dir / "best_model"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    for name in ["anchor.jpg", "mask.png", "profile.json", "evaluation.json"]:
        shutil.copy2(source_dir / name, best_dir / name)
    shutil.copy2(source_dir / "memory_bank.npz", best_dir / "model.npz")
    shutil.copy2(source_dir / "memory_bank.npz", best_dir / "memory_bank.npz")

    profile = json.loads((best_dir / "profile.json").read_text())
    evaluation = json.loads((best_dir / "evaluation.json").read_text())
    thresholds = {
        "anomaly_threshold": profile["anomaly_threshold"],
        "heatmap_threshold": profile["heatmap_threshold"],
        "min_region_area": profile["min_region_area"],
    }
    benchmark = {
        "target": target,
        "runtime": "torchvision-pytorch" if profile["feature_backend"] == "resnet18" else "opencv-numpy",
        "requires_pytorch_in_inference": profile["feature_backend"] == "resnet18",
        "estimated_ram_mb": evaluation["metrics"]["estimated_ram_mb"],
        "avg_latency_s": evaluation["metrics"]["avg_latency_s"],
        "target_latency_s": evaluation["metrics"]["target_latency_s"],
    }
    best_profile = {
        "version": MODEL_SUITE_VERSION,
        "family": family,
        "zone_id": zone_id,
        "target": target,
        "selected_candidate": selected_id,
        "production_ready": report["production_ready"],
        "artifact_type": "patchcore_embedding_npz",
        "model_file": "model.npz",
    }
    (best_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    (best_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (best_dir / "best_profile.json").write_text(json.dumps(best_profile, indent=2) + "\n")
    return {"best_model_dir": str(best_dir), **best_profile}


def inspect_best_model(
    family: str,
    zone_id: str,
    images: list[str | Path],
    registry_dir: str | Path = "data/model_registry",
    evidence_dir: str | Path = "reports/best_model_evidence",
    out: str | Path | None = None,
) -> list[dict[str, Any]]:
    best_dir = _suite_dir(registry_dir, family, zone_id) / "best_model"
    reports = inspect_anomaly_model_dir(
        model_dir=best_dir,
        family=family,
        zone_id=zone_id,
        images=images,
        evidence_dir=evidence_dir,
    )
    best_profile = json.loads((best_dir / "best_profile.json").read_text())
    for report in reports:
        result = report["result"]
        result["model_id"] = best_profile["selected_candidate"]
        result["target"] = best_profile["target"]
        result["production_ready"] = best_profile["production_ready"]
        if result["status"] == "correct" and not best_profile["production_ready"]:
            result["status"] = "review"
            result["message"] = "El mejor modelo no esta marcado como production_ready; requiere revision."
    if out:
        write_anomaly_report(out, reports)
    return reports


def _evaluate_candidate(
    candidate_dir: Path,
    spec: CandidateSpec,
    family: str,
    zone_id: str,
    rows: list[dict[str, str]],
    evidence_dir: Path,
    max_false_accept_rate: float,
) -> dict[str, Any]:
    predictions = []
    latencies = []
    for row in rows:
        start = time.perf_counter()
        report = inspect_anomaly_model_dir(
            model_dir=candidate_dir,
            family=family,
            zone_id=zone_id,
            images=[row["image_path"]],
            evidence_dir=evidence_dir,
        )[0]
        latencies.append(time.perf_counter() - start)
        predictions.append(
            {
                "image_path": row["image_path"],
                "label": row["label"],
                "status": report["result"]["status"],
                "anomaly_score": report["result"]["anomaly_score"],
            }
        )

    ok_rows = [item for item in predictions if item["label"] == "ok"]
    fault_rows = [item for item in predictions if item["label"] == "fault"]
    false_accepts = [item for item in fault_rows if item["status"] == "correct"]
    false_rejects = [item for item in ok_rows if item["status"] != "correct"]
    detected_faults = [item for item in fault_rows if item["status"] != "correct"]
    false_accept_rate = len(false_accepts) / len(fault_rows) if fault_rows else 1.0
    fault_recall = len(detected_faults) / len(fault_rows) if fault_rows else 0.0
    false_reject_rate = len(false_rejects) / len(ok_rows) if ok_rows else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    estimated_ram = _estimate_candidate_ram_mb(candidate_dir)
    production_ready = (
        bool(fault_rows)
        and false_accept_rate <= max_false_accept_rate
        and avg_latency <= spec.target_latency_s
        and estimated_ram <= spec.max_ram_mb
    )
    return {
        "samples": len(predictions),
        "ok_samples": len(ok_rows),
        "fault_samples": len(fault_rows),
        "false_accepts": len(false_accepts),
        "false_rejects": len(false_rejects),
        "false_accept_rate": round(false_accept_rate, 6),
        "fault_recall": round(fault_recall, 6),
        "false_reject_rate": round(false_reject_rate, 6),
        "avg_latency_s": round(avg_latency, 6),
        "target_latency_s": spec.target_latency_s,
        "estimated_ram_mb": estimated_ram,
        "max_ram_mb": spec.max_ram_mb,
        "production_ready": production_ready,
        "predictions": predictions,
    }


def _candidate_sort_key(candidate: dict[str, Any]):
    metrics = candidate["metrics"]
    return (
        metrics["false_accept_rate"],
        -metrics["fault_recall"],
        metrics["false_reject_rate"],
        metrics["avg_latency_s"],
        metrics["estimated_ram_mb"],
    )


def _load_manifest(path: str | Path, family: str, zone_id: str) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        if row.get("family") and row["family"] != family:
            continue
        if row.get("zone_id") and row["zone_id"] != zone_id:
            continue
        image_path = row.get("image_path") or row.get("path")
        if not image_path:
            raise ValueError("Manifest requires image_path or path column")
        label = _normalize_label(row.get("label") or row.get("state") or row.get("status") or "")
        normalized.append(
            {
                "image_path": image_path,
                "label": label,
                "mold_id": row.get("mold_id") or row.get("mold") or "unknown_mold",
                "session_id": row.get("session_id") or row.get("session") or image_path,
                "split": row.get("split", "").lower(),
            }
        )
    if not normalized:
        raise ValueError("No manifest rows matched the requested family/zone")
    return normalized


def _normalize_label(value: str) -> str:
    value = value.strip().lower()
    if value in {"ok", "correct", "normal", "pass", "good"}:
        return "ok"
    if value in {"fault", "incorrect", "fail", "bad", "simulated_fault", "defect"}:
        return "fault"
    raise ValueError(f"Unsupported label: {value!r}; use ok or fault")


def _ensure_splits(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if any(row["split"] for row in rows):
        return [dict(row, split=row["split"] or "train") for row in rows]

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["mold_id"], row["session_id"]), []).append(row)
    ordered_groups = list(groups.values())
    if len(ordered_groups) == 1:
        return [dict(row, split="train") for row in rows]

    split_rows = []
    for index, group in enumerate(ordered_groups):
        if index == 0:
            split = "train"
        elif index == len(ordered_groups) - 1:
            split = "test"
        else:
            split = "val"
        split_rows.extend(dict(row, split=split) for row in group)
    return split_rows


def _estimate_candidate_ram_mb(candidate_dir: Path) -> int:
    total_bytes = 0
    for path in candidate_dir.glob("*"):
        if path.is_file():
            total_bytes += path.stat().st_size
    return max(64, int(total_bytes / (1024 * 1024) * 4) + 64)


def _candidate_specs(target: str) -> list[CandidateSpec]:
    if target == "raspberry-pi":
        return RASPBERRY_CANDIDATES
    if target == "cloud-gpu":
        return CLOUD_GPU_CANDIDATES
    raise ValueError("target must be raspberry-pi or cloud-gpu")


def _suite_dir(registry_dir: str | Path, family: str, zone_id: str) -> Path:
    return Path(registry_dir) / family / zone_id


def _cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    while path != stop_at and path.exists():
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent
