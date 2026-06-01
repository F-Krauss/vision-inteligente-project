from __future__ import annotations

from pathlib import Path
from typing import Any

from mold_inspection.mold_segmenter import normalize_mold_crop
from mold_inspection.piece_inspector import inspect_expected_pieces

from .config import CloudSettings
from .schemas import InspectionCreateRequest, InspectionRecord
from .storage import ObjectStorage
from .store import MetadataStore

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class CloudInspectionPipeline:
    def __init__(self, settings: CloudSettings, store: MetadataStore, objects: ObjectStorage):
        self.settings = settings
        self.store = store
        self.objects = objects

    def inspect(self, request: InspectionCreateRequest) -> InspectionRecord:
        image_path = self.objects.materialize(request.image_uri)
        quality = evaluate_capture_quality(image_path)
        if not quality["ok"]:
            record = InspectionRecord(
                family=request.family,
                zone_id=request.zone_id,
                image_uri=request.image_uri,
                mold_id=request.mold_id,
                session_id=request.session_id,
                operator_id=request.operator_id,
                status="retake_photo",
                message=quality["message"],
                result={"capture_quality": quality},
                guidance=quality["guidance"],
                **_public_fields(request),
            )
            self.store.put("inspections", record.id, record.model_dump())
            return record

        normalized_path = self.settings.local_state_dir / "preprocessed" / request.family / request.zone_id / f"{Path(image_path).stem}.jpg"
        try:
            normalized_path, segmentation = normalize_mold_crop(image_path, normalized_path)
        except ValueError as exc:
            record = InspectionRecord(
                family=request.family,
                zone_id=request.zone_id,
                image_uri=request.image_uri,
                mold_id=request.mold_id,
                session_id=request.session_id,
                operator_id=request.operator_id,
                status="retake_photo",
                message=str(exc),
                result={"capture_quality": quality, "mold_segmentation": {"ok": False, "message": str(exc)}},
                guidance=["Toma otra foto con el molde completo y separado del fondo."],
                **_public_fields(request),
            )
            self.store.put("inspections", record.id, record.model_dump())
            return record
        if not segmentation.ok:
            record = InspectionRecord(
                family=request.family,
                zone_id=request.zone_id,
                image_uri=request.image_uri,
                mold_id=request.mold_id,
                session_id=request.session_id,
                operator_id=request.operator_id,
                status="retake_photo",
                message=segmentation.message,
                result={"capture_quality": quality, "mold_segmentation": segmentation.to_dict()},
                guidance=segmentation.guidance or ["Toma otra foto con el molde centrado y completo."],
                **_public_fields(request, segmentation=segmentation.to_dict()),
            )
            self.store.put("inspections", record.id, record.model_dump())
            return record

        best_model_dir = self.settings.model_registry_dir / request.family / request.zone_id / "best_model"
        if not (best_model_dir / "profile.json").exists():
            record = InspectionRecord(
                family=request.family,
                zone_id=request.zone_id,
                image_uri=request.image_uri,
                mold_id=request.mold_id,
                session_id=request.session_id,
                operator_id=request.operator_id,
                status="review",
                message="No hay modelo productivo para esta familia/zona; requiere entrenamiento o revision humana.",
                result={
                    "capture_quality": quality,
                    "mold_segmentation": segmentation.to_dict(),
                    "model_version": None,
                    "reason": "missing_best_model",
                },
                guidance=[
                    "Sube dataset ok/fault para esta zona.",
                    "Entrena y promueve una version de modelo antes de aprobar automaticamente.",
                ],
                **_public_fields(request, segmentation=segmentation.to_dict()),
            )
            self.store.put("inspections", record.id, record.model_dump())
            return record

        from mold_inspection.model_suite import inspect_best_model

        reports = inspect_best_model(
            family=request.family,
            zone_id=request.zone_id,
            images=[normalized_path],
            registry_dir=self.settings.model_registry_dir,
            evidence_dir=self.settings.evidence_dir,
        )
        result = reports[0]["result"]
        result["capture_quality"] = quality
        result["mold_segmentation"] = segmentation.to_dict()
        result["preprocessed_image_path"] = str(normalized_path)
        piece_inspection = inspect_expected_pieces(
            family=request.family,
            zone_id=request.zone_id,
            image_path=normalized_path,
            datasets=self.store.list("datasets"),
            registry_dir=self.settings.model_registry_dir,
        )
        result["piece_inspection"] = piece_inspection
        if result["status"] == "correct" and piece_inspection["status"] == "review":
            result["status"] = "review"
            result["message"] = piece_inspection["message"]
        evidence = _evidence_urls(result, self.settings.evidence_dir)
        record = InspectionRecord(
            family=request.family,
            zone_id=request.zone_id,
            image_uri=request.image_uri,
            mold_id=request.mold_id,
            session_id=request.session_id,
            operator_id=request.operator_id,
            status=result["status"],
            message=result["message"],
            result=result,
            guidance=_guidance_for_result(result),
            evidence=evidence,
            **_public_fields(request, segmentation=segmentation.to_dict(), result=result, evidence=evidence),
        )
        self.store.put("inspections", record.id, record.model_dump())
        return record


def evaluate_capture_quality(image_path: str | Path) -> dict[str, Any]:
    if cv2 is None:
        return {
            "ok": True,
            "message": "OpenCV no esta instalado; validacion de calidad limitada.",
            "guidance": [],
        }
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "ok": False,
            "message": "No se pudo leer la imagen.",
            "guidance": ["Vuelve a tomar la foto o sube un archivo JPEG/PNG/HEIC valido."],
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    height, width = gray.shape[:2]
    guidance: list[str] = []
    if min(width, height) < 480:
        guidance.append("Acercate menos o usa mayor resolucion; la zona quedo con pocos pixeles.")
    if blur_score < 18.0:
        guidance.append("Foto borrosa: limpia el lente, estabiliza el telefono y toma otra foto.")
    if brightness < 35.0:
        guidance.append("Iluminacion insuficiente: agrega luz frontal o evita sombras.")
    if brightness > 225.0:
        guidance.append("Brillo excesivo: evita reflejos directos sobre el molde.")
    return {
        "ok": not guidance,
        "message": "Calidad de captura suficiente." if not guidance else guidance[0],
        "guidance": guidance,
        "blur_score": round(blur_score, 3),
        "brightness": round(brightness, 3),
        "width": int(width),
        "height": int(height),
    }


def _evidence_urls(result: dict[str, Any], evidence_dir: Path) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key in ["aligned_image", "heatmap_image", "overlay_image"]:
        value = result.get(key)
        if not value:
            continue
        path = Path(value)
        try:
            rel = path.resolve().relative_to(evidence_dir.resolve())
            evidence[key] = f"/evidence/{rel.as_posix()}"
        except ValueError:
            evidence[key] = str(value)
    return evidence


def _guidance_for_result(result: dict[str, Any]) -> list[str]:
    if result["status"] == "correct":
        return ["Zona validada contra el modelo productivo."]
    if result["status"] == "retake_photo":
        return [result.get("message") or "Toma otra foto siguiendo la guia de captura."]
    guidance = ["Requiere revision humana antes de aprobar."]
    if result.get("difference_regions"):
        guidance.append("Revisa las regiones marcadas en el overlay.")
    return guidance


def _public_fields(
    request: InspectionCreateRequest,
    segmentation: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    missing_regions = _missing_regions(result or {})
    return {
        "identified_mold": request.mold_id or request.family,
        "identified_zone": request.zone_id,
        "confidence": _confidence(segmentation, result),
        "mold_polygon": segmentation.get("polygon_normalized", []) if segmentation else [],
        "missing_regions": missing_regions,
        "overlay_image_uri": (evidence or {}).get("overlay_image"),
    }


def _confidence(segmentation: dict[str, Any] | None, result: dict[str, Any] | None) -> float | None:
    if result and isinstance(result.get("confidence"), (int, float)):
        return float(result["confidence"])
    if segmentation and isinstance(segmentation.get("confidence"), (int, float)):
        return float(segmentation["confidence"])
    return None


def _missing_regions(result: dict[str, Any]) -> list[list[dict[str, float]]]:
    findings = ((result.get("piece_inspection") or {}).get("findings") or [])
    regions: list[list[dict[str, float]]] = []
    for finding in findings:
        if finding.get("status") != "missing":
            continue
        region = finding.get("region") or finding.get("polygon_normalized") or finding.get("polygon")
        if isinstance(region, list) and region:
            regions.append(region)
    return regions
