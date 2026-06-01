from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import shutil
import subprocess
import tempfile

try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover - exercised through CLI runtime checks.
    cv2 = None
    np = None
    structural_similarity = None


@dataclass(frozen=True)
class CaptureQuality:
    ok: bool
    message: str
    matched_keypoints: int
    inlier_ratio: float
    brightness_delta: float
    blur_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "matched_keypoints": self.matched_keypoints,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "brightness_delta": round(self.brightness_delta, 4),
            "blur_score": round(self.blur_score, 4),
        }


@dataclass(frozen=True)
class DifferenceRegion:
    x: int
    y: int
    width: int
    height: int
    area: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass(frozen=True)
class GoldenInspectionResult:
    status: str
    message: str
    similarity: float | None
    quality: CaptureQuality
    difference_regions: list[DifferenceRegion]
    aligned_image: str | None
    diff_image: str | None
    matched_reference: str | None = None
    difference_area_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "matched_reference": self.matched_reference,
            "difference_area_ratio": round(self.difference_area_ratio, 4),
            "quality": self.quality.as_dict(),
            "difference_regions": [region.as_dict() for region in self.difference_regions],
            "aligned_image": self.aligned_image,
            "diff_image": self.diff_image,
        }


def create_reference(
    source_image: str | Path,
    family: str,
    zone_id: str,
    reference_id: str = "default",
    references_dir: str | Path = "data/references",
) -> Path:
    _require_vision_deps()
    source_image = Path(source_image)
    destination = Path(references_dir) / family / zone_id / f"{_safe_reference_id(reference_id)}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_as_jpeg(source_image, destination)
    return destination


def inspect_against_reference(
    image_path: str | Path,
    family: str,
    zone_id: str,
    references_dir: str | Path = "data/references",
    evidence_dir: str | Path = "reports/evidence",
    min_similarity: float = 0.90,
    min_keypoints: int = 40,
    min_inlier_ratio: float = 0.18,
    max_brightness_delta: float = 65.0,
    min_blur_score: float = 35.0,
    difference_threshold: int = 38,
    min_region_area: int = 350,
    min_comparable_similarity: float = 0.70,
    max_difference_area_ratio: float = 0.60,
) -> GoldenInspectionResult:
    _require_vision_deps()
    image_path = Path(image_path)
    reference_paths = _reference_paths(references_dir, family, zone_id)
    if not reference_paths:
        raise FileNotFoundError(f"Missing reference images for: {Path(references_dir) / family / zone_id}")

    candidate = _read_color(image_path)
    attempts: list[tuple[Path, Any, Any, CaptureQuality, Any]] = []

    for reference_path in reference_paths:
        reference = _read_color(reference_path)
        aligned, quality, valid_mask = _align_candidate(
            reference,
            candidate,
            min_keypoints=min_keypoints,
            min_inlier_ratio=min_inlier_ratio,
            max_brightness_delta=max_brightness_delta,
            min_blur_score=min_blur_score,
        )
        attempts.append((reference_path, reference, aligned, quality, valid_mask))

    valid_attempts = [attempt for attempt in attempts if attempt[3].ok]
    if not valid_attempts:
        best_failed = _best_failed_attempt(attempts)
        quality = best_failed[3]
        return GoldenInspectionResult(
            status="retake_photo",
            message=quality.message,
            similarity=None,
            quality=quality,
            difference_regions=[],
            aligned_image=None,
            diff_image=None,
            matched_reference=None,
        )

    scored_attempts = []
    for reference_path, reference, aligned, quality, valid_mask in valid_attempts:
        similarity, diff_map = _similarity(reference, aligned, valid_mask)
        scored_attempts.append((similarity, reference_path, reference, aligned, quality, diff_map, valid_mask))

    similarity, reference_path, reference, aligned, quality, diff_map, valid_mask = max(
        scored_attempts,
        key=lambda attempt: (attempt[0], attempt[4].inlier_ratio),
    )

    regions, mask = _difference_regions(diff_map, difference_threshold, min_region_area, valid_mask)
    difference_area_ratio = _difference_area_ratio(regions, valid_mask, reference.shape[:2])

    evidence_base = Path(evidence_dir) / family / zone_id / image_path.stem
    evidence_base.mkdir(parents=True, exist_ok=True)
    reference_slug = reference_path.stem
    aligned_path = evidence_base / f"aligned_{reference_slug}.jpg"
    diff_path = evidence_base / f"diff_{reference_slug}.jpg"
    cv2.imwrite(str(aligned_path), aligned)
    cv2.imwrite(str(diff_path), _draw_regions(aligned, regions, mask))

    if similarity < min_comparable_similarity or difference_area_ratio > max_difference_area_ratio:
        status = "retake_photo"
        message = "La foto no es suficientemente comparable con las referencias. Tome otra foto mas parecida a una muestra golden."
    elif similarity < min_similarity or regions:
        status = "review"
        message = "Diferencias contra referencia detectadas; requiere revision."
    else:
        status = "correct"
        message = "La zona coincide con la referencia dentro de tolerancia."

    return GoldenInspectionResult(
        status=status,
        message=message,
        similarity=similarity,
        quality=quality,
        difference_regions=regions,
        aligned_image=str(aligned_path),
        diff_image=str(diff_path),
        matched_reference=reference_path.stem,
        difference_area_ratio=difference_area_ratio,
    )


def write_golden_report(path: str | Path, reports: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reports": reports}, indent=2) + "\n")


def _reference_paths(references_dir: str | Path, family: str, zone_id: str) -> list[Path]:
    base = Path(references_dir) / family / zone_id
    paths: list[Path] = []
    if base.is_dir():
        paths.extend(sorted(path for path in base.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}))

    legacy_path = Path(references_dir) / family / f"{zone_id}.jpg"
    if legacy_path.exists():
        paths.append(legacy_path)

    return paths


def _best_failed_attempt(attempts: list[tuple[Path, Any, Any, CaptureQuality, Any]]) -> tuple[Path, Any, Any, CaptureQuality, Any]:
    return max(
        attempts,
        key=lambda attempt: (
            attempt[3].matched_keypoints,
            attempt[3].inlier_ratio,
            -attempt[3].brightness_delta,
            attempt[3].blur_score,
        ),
    )


def _align_candidate(
    reference,
    candidate,
    min_keypoints: int,
    min_inlier_ratio: float,
    max_brightness_delta: float,
    min_blur_score: float,
) -> tuple[Any, CaptureQuality, Any]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cand_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    brightness_delta = abs(float(ref_gray.mean()) - float(cand_gray.mean()))
    blur_score = float(cv2.Laplacian(cand_gray, cv2.CV_64F).var())

    if blur_score < min_blur_score:
        return candidate, CaptureQuality(
            ok=False,
            message="Foto borrosa. Tome otra foto con el celular estable y enfoque claro.",
            matched_keypoints=0,
            inlier_ratio=0.0,
            brightness_delta=brightness_delta,
            blur_score=blur_score,
        ), None

    if brightness_delta > max_brightness_delta:
        return candidate, CaptureQuality(
            ok=False,
            message="Iluminacion muy distinta a la referencia. Tome otra foto con luz mas uniforme.",
            matched_keypoints=0,
            inlier_ratio=0.0,
            brightness_delta=brightness_delta,
            blur_score=blur_score,
        ), None

    orb = cv2.ORB_create(nfeatures=6000)
    ref_keypoints, ref_desc = orb.detectAndCompute(ref_gray, None)
    cand_keypoints, cand_desc = orb.detectAndCompute(cand_gray, None)

    if ref_desc is None or cand_desc is None:
        return candidate, CaptureQuality(
            ok=False,
            message="No se encontraron suficientes puntos de referencia. Repita la foto desde la zona correcta.",
            matched_keypoints=0,
            inlier_ratio=0.0,
            brightness_delta=brightness_delta,
            blur_score=blur_score,
        ), None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(ref_desc, cand_desc), key=lambda item: item.distance)
    good_matches = matches[: min(350, len(matches))]

    if len(good_matches) < min_keypoints:
        return candidate, CaptureQuality(
            ok=False,
            message="La foto no coincide con la zona o angulo esperado. Tome otra foto siguiendo la guia.",
            matched_keypoints=len(good_matches),
            inlier_ratio=0.0,
            brightness_delta=brightness_delta,
            blur_score=blur_score,
        ), None

    ref_points = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    cand_points = np.float32([cand_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(cand_points, ref_points, cv2.RANSAC, 5.0)

    if homography is None or inlier_mask is None:
        inlier_ratio = 0.0
    else:
        inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))

    if homography is None or inlier_ratio < min_inlier_ratio:
        return candidate, CaptureQuality(
            ok=False,
            message="Perspectiva o encuadre fuera de tolerancia. Tome otra foto mas parecida a la referencia.",
            matched_keypoints=len(good_matches),
            inlier_ratio=inlier_ratio,
            brightness_delta=brightness_delta,
            blur_score=blur_score,
        ), None

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(candidate, homography, (width, height))
    source_mask = np.full(candidate.shape[:2], 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(source_mask, homography, (width, height))
    valid_mask = cv2.threshold(valid_mask, 250, 255, cv2.THRESH_BINARY)[1]
    valid_mask = _erode_valid_mask(valid_mask)
    return aligned, CaptureQuality(
        ok=True,
        message="Foto valida para comparar.",
        matched_keypoints=len(good_matches),
        inlier_ratio=inlier_ratio,
        brightness_delta=brightness_delta,
        blur_score=blur_score,
    ), valid_mask


def _similarity(reference, aligned, valid_mask=None) -> tuple[float, Any]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    score, diff = structural_similarity(ref_gray, aligned_gray, full=True)
    diff_map = ((1.0 - diff) * 255).astype("uint8")
    if valid_mask is None:
        return float(score), diff_map

    valid_pixels = valid_mask > 0
    if not np.any(valid_pixels):
        return 0.0, diff_map

    masked_diff = np.zeros_like(diff_map)
    masked_diff[valid_pixels] = diff_map[valid_pixels]
    masked_score = float(np.mean(diff[valid_pixels]))
    return masked_score, masked_diff


def _difference_regions(diff_map, threshold: int, min_region_area: int, valid_mask=None) -> tuple[list[DifferenceRegion], Any]:
    _, mask = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)
    if valid_mask is not None:
        mask = cv2.bitwise_and(mask, valid_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[DifferenceRegion] = []

    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_region_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append(DifferenceRegion(x=x, y=y, width=width, height=height, area=area))

    regions.sort(key=lambda region: region.area, reverse=True)
    return regions, mask


def _difference_area_ratio(regions: list[DifferenceRegion], valid_mask, image_shape: tuple[int, int]) -> float:
    difference_area = float(sum(region.area for region in regions))
    if valid_mask is not None and np.any(valid_mask > 0):
        denominator = float(np.count_nonzero(valid_mask))
    else:
        denominator = float(image_shape[0] * image_shape[1])
    if denominator <= 0:
        return 0.0
    return min(1.0, difference_area / denominator)


def _erode_valid_mask(valid_mask):
    height, width = valid_mask.shape[:2]
    kernel_size = max(7, int(min(height, width) * 0.02))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(valid_mask, kernel, iterations=1)


def _draw_regions(image, regions: list[DifferenceRegion], mask):
    overlay = image.copy()
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.72, heat, 0.28, 0)
    for region in regions:
        cv2.rectangle(
            overlay,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            (0, 0, 255),
            3,
        )
    return overlay


def _read_color(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    converted = _convert_with_sips(path)
    if converted is not None:
        image = cv2.imread(str(converted), cv2.IMREAD_COLOR)
        converted.unlink(missing_ok=True)
        if image is not None:
            return image

    raise ValueError(f"Could not read image: {path}")


def _copy_as_jpeg(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is not None:
        cv2.imwrite(str(destination), image)
        return
    converted = _convert_with_sips(source)
    if converted is not None:
        shutil.copy2(converted, destination)
        converted.unlink(missing_ok=True)
        return
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, destination)
        return
    raise ValueError(f"Could not convert image to JPEG: {source}")


def _convert_with_sips(source: Path) -> Path | None:
    if source.suffix.lower() not in {".heic", ".heif", ".png", ".tif", ".tiff"}:
        return None
    fd, destination_name = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    destination = Path(destination_name)
    try:
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(source), "--out", str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return destination
    except (FileNotFoundError, subprocess.CalledProcessError):
        destination.unlink(missing_ok=True)
        return None


def _safe_reference_id(reference_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in reference_id.strip())
    return safe or "default"


def _require_vision_deps() -> None:
    if cv2 is None or np is None or structural_similarity is None:
        raise RuntimeError('Install vision dependencies: python3 -m pip install -e ".[vision]"')
