from __future__ import annotations

from pathlib import Path
from typing import Any

from mold_inspection.mold_segmenter import normalize_mold_crop
from mold_inspection.piece_inspector import (
    inspect_expected_pieces,
    inspect_expected_pieces_against_reference,
    inspect_expected_pieces_against_references,
)

from .config import CloudSettings
from .cv_inspector import inspect_with_cv_consensus
from .model_artifacts import materialize_model_uri
from .references import (
    expected_pieces_for_zone,
    gather_annotated_references,
    get_zone_reference,
    get_zone_references,
)
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
        segmentation_warning = segmentation.message if not segmentation.ok else None
        if not segmentation.ok and not _can_use_comparable_segmentation(segmentation.to_dict()):
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

        expected_pieces = expected_pieces_for_zone(request.zone_id, self.store, family=request.family)
        best_model_dir = self.settings.model_registry_dir / request.family / request.zone_id / "best_model"
        piece_detector_path = self._piece_detector_path(request)
        has_anomaly_model = (best_model_dir / "profile.json").exists()
        if not has_anomaly_model and not piece_detector_path:
            result = {
                "capture_quality": quality,
                "mold_segmentation": segmentation.to_dict(),
                "model_version": None,
                "reason": "missing_best_model",
            }
            references = get_zone_references(request.zone_id, self.store, family=request.family)
            reference = references[-1] if references else None
            piece_inspection = self._inspect_reference_pieces(
                request=request,
                normalized_path=normalized_path,
                expected_pieces=expected_pieces,
                references=references,
            )
            if piece_inspection:
                result["reason"] = "reference_roi_diff_without_model"
                result["piece_inspection"] = piece_inspection
                result["reference"] = {
                    "id": reference.get("id") if reference else None,
                    "reference_id": reference.get("reference_id") if reference else None,
                    "image_uri": reference.get("image_uri") if reference else None,
                }
                overlay_image = piece_inspection.get("overlay_image")
                if overlay_image:
                    result["overlay_image"] = overlay_image
                if segmentation_warning:
                    result["capture_warning"] = segmentation_warning
                status = piece_inspection["status"] if piece_inspection["status"] in {"correct", "review"} else "review"
                message = piece_inspection["message"]
                guidance = _guidance_for_result({"status": status, "message": message, "piece_inspection": piece_inspection})
                if status == "correct":
                    guidance = ["Zona validada contra referencia golden.", "Entrena modelo productivo para aprobacion automatica robusta."]
                else:
                    guidance.append("Confirma piezas marcadas y agrega esta foto al dataset.")
                evidence = _evidence_urls(result, self.settings.evidence_dir)
                record = InspectionRecord(
                    family=request.family,
                    zone_id=request.zone_id,
                    image_uri=request.image_uri,
                    mold_id=request.mold_id,
                    session_id=request.session_id,
                    operator_id=request.operator_id,
                    status=status,
                    message=message,
                    result=result,
                    guidance=guidance,
                    evidence=evidence,
                    **_public_fields(request, segmentation=segmentation.to_dict(), result=result, evidence=evidence),
                )
                self.store.put("inspections", record.id, record.model_dump())
                return record

            record = InspectionRecord(
                family=request.family,
                zone_id=request.zone_id,
                image_uri=request.image_uri,
                mold_id=request.mold_id,
                session_id=request.session_id,
                operator_id=request.operator_id,
                status="review",
                message="No hay modelo productivo para esta familia/zona; requiere entrenamiento o revision humana.",
                result=result,
                guidance=[
                    "Sube dataset ok/fault para esta zona.",
                    "Entrena y promueve una version de modelo antes de aprobar automaticamente.",
                ],
                **_public_fields(request, segmentation=segmentation.to_dict()),
            )
            self.store.put("inspections", record.id, record.model_dump())
            return record

        if has_anomaly_model:
            from mold_inspection.model_suite import inspect_best_model

            reports = inspect_best_model(
                family=request.family,
                zone_id=request.zone_id,
                images=[normalized_path],
                registry_dir=self.settings.model_registry_dir,
                evidence_dir=self.settings.evidence_dir,
            )
            result = reports[0]["result"]
        else:
            result = {
                "status": "correct",
                "message": "Modelo YOLO de piezas disponible; anomaly guardrail pendiente.",
                "model_version": None,
                "reason": "piece_detector_without_anomaly_model",
            }
        result["capture_quality"] = quality
        result["mold_segmentation"] = segmentation.to_dict()
        result["preprocessed_image_path"] = str(normalized_path)
        piece_inspection = inspect_expected_pieces(
            family=request.family,
            zone_id=request.zone_id,
            image_path=normalized_path,
            datasets=self.store.list("datasets"),
            registry_dir=self.settings.model_registry_dir,
            expected_pieces=expected_pieces,
            detector_path=piece_detector_path,
        )
        result["piece_inspection"] = piece_inspection
        if result["status"] == "correct" and piece_inspection["status"] == "review":
            result["status"] = "review"
            result["message"] = piece_inspection["message"]

        # ── Agreement gate ────────────────────────────────────────────────────
        # The trained model (anomaly ∧ YOLO pieces) only auto-approves when the
        # reference cross-check agrees too. Each signal is independent, so requiring
        # consensus before "correct" drives false-approval toward zero; any
        # disagreement defers to human review (the project's standing priority).
        if result["status"] == "correct":
            cross_check = self._reference_cross_check(request)
            if cross_check:
                result["reference_cross_check"] = cross_check
                if cross_check.get("status") == "review":
                    result["status"] = "review"
                    result["message"] = cross_check.get("message") or result["message"]

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

    def _piece_detector_path(self, request: InspectionCreateRequest) -> Path | None:
        local = self.settings.model_registry_dir / request.family / request.zone_id / "piece_detector" / "best.pt"
        if local.exists():
            return local
        model_id = f"best_{request.family}_{request.zone_id}".replace("/", "_")
        version = self.store.get("model_versions", model_id) or {}
        model_uri = str(version.get("model_uri") or "")
        if not model_uri or not model_uri.endswith(".pt"):
            return None
        try:
            return materialize_model_uri(model_uri, self.settings, filename="best.pt")
        except Exception:
            return None

    def _annotated_references_local(self, request: InspectionCreateRequest) -> list[dict[str, Any]]:
        """Annotated golden images for the zone with their URIs materialized to local
        paths (the per-part diff reads them with cv2.imread). Unreadable/HEIC refs are
        dropped — the consensus uses whatever readable references remain."""
        annotated_local: list[dict[str, Any]] = []
        for ref in gather_annotated_references(request.zone_id, self.store, family=request.family):
            try:
                local_path = self.objects.materialize(str(ref["image_uri"]))
            except Exception:
                continue
            annotated_local.append({"image_path": str(local_path), "boxes": ref["boxes"]})
        return annotated_local

    def _reference_cross_check(self, request: InspectionCreateRequest) -> dict[str, Any] | None:
        """Independent reference verdict used to gate the trained model's auto-approval.
        Prefers the per-part annotated-reference consensus; falls back to the coarse
        golden-reference CV consensus. Returns None when the zone has no references to
        cross-check against (nothing to disagree with)."""
        annotated = self._annotated_references_local(request)
        if annotated:
            return inspect_expected_pieces_against_references(
                family=request.family,
                zone_id=request.zone_id,
                candidate_image_path=str(self.objects.materialize(str(request.image_uri))),
                annotated_references=annotated,
                evidence_dir=self.settings.evidence_dir,
            )
        usable_refs = [
            ref for ref in get_zone_references(request.zone_id, self.store, family=request.family)
            if ref.get("image_uri")
        ]
        if not usable_refs:
            return None
        cand_uri = str(request.image_uri)
        cand_path: str | Path = cand_uri if cand_uri.startswith("gs://") else self.objects.materialize(cand_uri)
        ref_paths: list[str | Path] = [
            (uri if uri.startswith("gs://") else self.objects.materialize(uri))
            for uri in (str(ref["image_uri"]) for ref in usable_refs)
        ]
        try:
            return inspect_with_cv_consensus(
                reference_image_paths=ref_paths,
                candidate_image_path=cand_path,
                family=request.family,
                zone_id=request.zone_id,
                evidence_dir=self.settings.evidence_dir,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Reference cross-check failed: %s", exc)
            return None

    def _inspect_reference_pieces(
        self,
        request: InspectionCreateRequest,
        normalized_path: Path,
        expected_pieces: list[dict[str, Any]],
        references: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        usable_refs = [ref for ref in references if ref.get("image_uri")]

        # ── Multi-annotated-reference per-part consensus: primary when the zone has
        # annotated golden images. Checks each known part location across every
        # annotated reference (false-rejection-safe three-state vote). Most precise
        # for the tiny parts, since it inspects each ROI rather than a global blob. ─
        annotated = self._annotated_references_local(request)
        if annotated:
            consensus = inspect_expected_pieces_against_references(
                family=request.family,
                zone_id=request.zone_id,
                candidate_image_path=str(self.objects.materialize(str(request.image_uri))),
                annotated_references=annotated,
                evidence_dir=self.settings.evidence_dir,
            )
            if consensus and consensus.get("status") in {"correct", "review"}:
                return consensus

        if not usable_refs:
            return None

        # ── Classical CV consensus: coarse fallback ───────────────────────────
        # Compares the candidate against EVERY golden reference and keeps only
        # findings corroborated by a majority of references. Deterministic, no
        # API calls; single-reference phantoms (lighting/pose) are suppressed.
        cand_uri = str(request.image_uri)
        cand_path_cv: str | Path = (
            cand_uri if cand_uri.startswith("gs://") else self.objects.materialize(cand_uri)
        )
        ref_paths_cv: list[str | Path] = [
            (uri if uri.startswith("gs://") else self.objects.materialize(uri))
            for uri in (str(ref["image_uri"]) for ref in usable_refs)
        ]
        try:
            cv_result = inspect_with_cv_consensus(
                reference_image_paths=ref_paths_cv,
                candidate_image_path=cand_path_cv,
                family=request.family,
                zone_id=request.zone_id,
                evidence_dir=self.settings.evidence_dir,
            )
            if cv_result and cv_result.get("status") in {"correct", "review"}:
                return cv_result
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "CV consensus inspection error, falling back to pixel-diff: %s", exc
            )

        # ── Pixel-diff fallback (legacy): only reached if CV could not compare ─
        if not expected_pieces:
            return None
        primary_uri = str(usable_refs[-1]["image_uri"])
        try:
            reference_path = self.objects.materialize(primary_uri)
            reference_normalized_path = (
                self.settings.local_state_dir
                / "preprocessed"
                / request.family
                / request.zone_id
                / f"{Path(reference_path).stem}_reference.jpg"
            )
            reference_normalized_path, _ = normalize_mold_crop(reference_path, reference_normalized_path)
        except ValueError:
            reference_normalized_path = Path(reference_path)
        return inspect_expected_pieces_against_reference(
            family=request.family,
            zone_id=request.zone_id,
            image_path=normalized_path,
            reference_image_path=reference_normalized_path,
            expected_pieces=expected_pieces,
            evidence_dir=self.settings.evidence_dir,
        )


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


def _can_use_comparable_segmentation(segmentation: dict[str, Any]) -> bool:
    if float(segmentation.get("confidence") or 0.0) < 0.55:
        return False
    bbox = segmentation.get("bbox_normalized") or {}
    width = float(bbox.get("width") or 0.0)
    height = float(bbox.get("height") or 0.0)
    area = float(segmentation.get("mold_area_ratio") or (width * height))
    return width >= 0.45 and height >= 0.45 and area >= 0.18


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
