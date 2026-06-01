from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from .golden_reference import (
    CaptureQuality,
    DifferenceRegion,
    _align_candidate,
    _read_color,
)

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


ANOMALY_VERSION = 1


@dataclass(frozen=True)
class AnomalyInspectionResult:
    status: str
    message: str
    matched_reference: str | None
    anomaly_score: float | None
    anomaly_threshold: float | None
    quality: CaptureQuality
    difference_regions: list[DifferenceRegion]
    aligned_image: str | None
    heatmap_image: str | None
    overlay_image: str | None
    valid_mask_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "matched_reference": self.matched_reference,
            "anomaly_score": round(self.anomaly_score, 4) if self.anomaly_score is not None else None,
            "anomaly_threshold": round(self.anomaly_threshold, 4) if self.anomaly_threshold is not None else None,
            "valid_mask_ratio": round(self.valid_mask_ratio, 4),
            "quality": self.quality.as_dict(),
            "difference_regions": [region.as_dict() for region in self.difference_regions],
            "aligned_image": self.aligned_image,
            "heatmap_image": self.heatmap_image,
            "overlay_image": self.overlay_image,
        }


def set_zone_mask(
    family: str,
    zone_id: str,
    mask_path: str | Path,
    anomaly_dir: str | Path = "data/anomaly",
) -> Path:
    _require_deps()
    destination = _zone_dir(anomaly_dir, family, zone_id) / "mask.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask = _read_mask(mask_path)
    cv2.imwrite(str(destination), mask)
    return destination


def train_anomaly_model(
    family: str,
    zone_id: str,
    images: list[str | Path],
    mask_path: str | Path | None = None,
    anomaly_dir: str | Path = "data/anomaly",
    feature_backend: str = "resnet18",
    image_size: int = 512,
    max_bank_patches: int = 20000,
    location_weight: float = 0.15,
    anomaly_threshold: float | None = None,
    heatmap_threshold: float | None = None,
    min_keypoints: int = 40,
    min_inlier_ratio: float = 0.15,
    max_brightness_delta: float = 70.0,
    min_blur_score: float = 25.0,
    min_mask_coverage: float = 0.55,
    min_region_area: int = 350,
) -> dict[str, Any]:
    _require_deps()
    if not images:
        raise ValueError("train-anomaly requires at least one image")

    zone_dir = _zone_dir(anomaly_dir, family, zone_id)
    zone_dir.mkdir(parents=True, exist_ok=True)
    anchor = _read_color(Path(images[0]))
    anchor_path = zone_dir / "anchor.jpg"
    cv2.imwrite(str(anchor_path), anchor)

    if mask_path:
        mask = _read_mask(mask_path)
    else:
        existing_mask = zone_dir / "mask.png"
        mask = _read_mask(existing_mask) if existing_mask.exists() else np.full(anchor.shape[:2], 255, dtype=np.uint8)
    mask = _resize_mask(mask, anchor.shape[:2])
    mask_path_out = zone_dir / "mask.png"
    cv2.imwrite(str(mask_path_out), mask)

    extractor = PatchFeatureExtractor(
        backend=feature_backend,
        image_size=image_size,
        location_weight=location_weight,
    )

    training_items: list[tuple[str, Any, Any]] = []
    bank_parts: list[Any] = []
    for index, image_path in enumerate(images):
        image_path = Path(image_path)
        image = _read_color(image_path)
        if index == 0:
            aligned = anchor
            valid_mask = np.full(anchor.shape[:2], 255, dtype=np.uint8)
        else:
            aligned, quality, valid_mask = _align_candidate(
                anchor,
                image,
                min_keypoints=min_keypoints,
                min_inlier_ratio=min_inlier_ratio,
                max_brightness_delta=max_brightness_delta,
                min_blur_score=min_blur_score,
            )
            if not quality.ok:
                raise ValueError(f"Training image is not comparable: {image_path}: {quality.message}")

        combined_mask = _combine_masks(mask, valid_mask)
        coverage = _mask_coverage(mask, combined_mask)
        if coverage < min_mask_coverage:
            raise ValueError(f"Training image mask coverage too low: {image_path}: {coverage:.3f}")

        features, grid_mask, grid_shape = extractor.extract(aligned, combined_mask)
        selected = features[grid_mask.reshape(-1)]
        if selected.size == 0:
            raise ValueError(f"No usable patches inside mask: {image_path}")
        bank_parts.append(selected)
        training_items.append((str(image_path), aligned, combined_mask))

    memory_bank = np.concatenate(bank_parts, axis=0).astype("float32")
    memory_bank = _subsample_bank(memory_bank, max_bank_patches)

    train_scores: list[float] = []
    train_distances: list[Any] = []
    for _, aligned, combined_mask in training_items:
        features, grid_mask, grid_shape = extractor.extract(aligned, combined_mask)
        distances = _nearest_distances(features, memory_bank)
        masked_distances = distances[grid_mask.reshape(-1)]
        if masked_distances.size:
            train_scores.append(float(np.percentile(masked_distances, 99)))
            train_distances.append(masked_distances)

    default_anomaly, default_heatmap = _default_thresholds(feature_backend)
    all_train_distances = np.concatenate(train_distances) if train_distances else np.array([0.0], dtype="float32")
    anomaly_threshold = float(
        anomaly_threshold
        if anomaly_threshold is not None
        else max(default_anomaly, float(np.percentile(train_scores or [0.0], 95)) * 2.0)
    )
    heatmap_threshold = float(
        heatmap_threshold
        if heatmap_threshold is not None
        else max(default_heatmap, float(np.percentile(all_train_distances, 99)) * 2.0)
    )

    np.savez_compressed(
        zone_dir / "memory_bank.npz",
        features=memory_bank,
    )

    profile = {
        "version": ANOMALY_VERSION,
        "family": family,
        "zone_id": zone_id,
        "feature_backend": extractor.backend,
        "pretrained": extractor.pretrained,
        "image_size": image_size,
        "grid_shape": list(grid_shape),
        "location_weight": location_weight,
        "max_bank_patches": max_bank_patches,
        "memory_patches": int(memory_bank.shape[0]),
        "anomaly_threshold": anomaly_threshold,
        "heatmap_threshold": heatmap_threshold,
        "min_keypoints": min_keypoints,
        "min_inlier_ratio": min_inlier_ratio,
        "max_brightness_delta": max_brightness_delta,
        "min_blur_score": min_blur_score,
        "min_mask_coverage": min_mask_coverage,
        "min_region_area": min_region_area,
        "anchor_shape": [int(anchor.shape[0]), int(anchor.shape[1])],
        "training_images": [str(path) for path in images],
    }
    (zone_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    return profile


def inspect_anomaly_images(
    family: str,
    zone_id: str,
    images: list[str | Path],
    anomaly_dir: str | Path = "data/anomaly",
    evidence_dir: str | Path = "reports/anomaly_evidence",
    anomaly_threshold: float | None = None,
    heatmap_threshold: float | None = None,
    min_region_area: int | None = None,
) -> list[dict[str, Any]]:
    _require_deps()
    zone_dir = _zone_dir(anomaly_dir, family, zone_id)
    return inspect_anomaly_model_dir(
        model_dir=zone_dir,
        family=family,
        zone_id=zone_id,
        images=images,
        evidence_dir=evidence_dir,
        anomaly_threshold=anomaly_threshold,
        heatmap_threshold=heatmap_threshold,
        min_region_area=min_region_area,
    )


def inspect_anomaly_model_dir(
    model_dir: str | Path,
    family: str,
    zone_id: str,
    images: list[str | Path],
    evidence_dir: str | Path = "reports/anomaly_evidence",
    anomaly_threshold: float | None = None,
    heatmap_threshold: float | None = None,
    min_region_area: int | None = None,
) -> list[dict[str, Any]]:
    _require_deps()
    zone_dir = Path(model_dir)
    profile = json.loads((zone_dir / "profile.json").read_text())
    anchor = _read_color(zone_dir / "anchor.jpg")
    mask = _resize_mask(_read_mask(zone_dir / "mask.png"), anchor.shape[:2])
    bank_path = zone_dir / "memory_bank.npz"
    if not bank_path.exists():
        bank_path = zone_dir / "model.npz"
    memory_bank = np.load(bank_path)["features"].astype("float32")
    extractor = PatchFeatureExtractor(
        backend=profile["feature_backend"],
        image_size=int(profile["image_size"]),
        location_weight=float(profile["location_weight"]),
    )

    reports: list[dict[str, Any]] = []
    for image_path in images:
        result = _inspect_one(
            Path(image_path),
            family,
            zone_id,
            anchor,
            mask,
            memory_bank,
            profile,
            extractor,
            evidence_dir,
            anomaly_threshold=anomaly_threshold,
            heatmap_threshold=heatmap_threshold,
            min_region_area=min_region_area,
        )
        reports.append(
            {
                "image_path": str(image_path),
                "family": family,
                "zone_id": zone_id,
                "result": result.as_dict(),
            }
        )
    return reports


def write_anomaly_report(path: str | Path, reports: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"reports": reports}, indent=2) + "\n")


def _inspect_one(
    image_path: Path,
    family: str,
    zone_id: str,
    anchor,
    mask,
    memory_bank,
    profile: dict[str, Any],
    extractor: "PatchFeatureExtractor",
    evidence_dir: str | Path,
    anomaly_threshold: float | None,
    heatmap_threshold: float | None,
    min_region_area: int | None,
) -> AnomalyInspectionResult:
    image = _read_color(image_path)
    aligned, quality, valid_mask = _align_candidate(
        anchor,
        image,
        min_keypoints=int(profile["min_keypoints"]),
        min_inlier_ratio=float(profile["min_inlier_ratio"]),
        max_brightness_delta=float(profile["max_brightness_delta"]),
        min_blur_score=float(profile["min_blur_score"]),
    )

    if not quality.ok:
        return AnomalyInspectionResult(
            status="retake_photo",
            message=quality.message,
            matched_reference="anchor",
            anomaly_score=None,
            anomaly_threshold=anomaly_threshold or float(profile["anomaly_threshold"]),
            quality=quality,
            difference_regions=[],
            aligned_image=None,
            heatmap_image=None,
            overlay_image=None,
        )

    combined_mask = _combine_masks(mask, valid_mask)
    coverage = _mask_coverage(mask, combined_mask)
    if coverage < float(profile["min_mask_coverage"]):
        return AnomalyInspectionResult(
            status="retake_photo",
            message="La mascara util queda demasiado fuera del encuadre. Tome otra foto de la zona completa.",
            matched_reference="anchor",
            anomaly_score=None,
            anomaly_threshold=anomaly_threshold or float(profile["anomaly_threshold"]),
            quality=quality,
            difference_regions=[],
            aligned_image=None,
            heatmap_image=None,
            overlay_image=None,
            valid_mask_ratio=coverage,
        )

    features, grid_mask, grid_shape = extractor.extract(aligned, combined_mask)
    distances = _nearest_distances(features, memory_bank)
    grid_distances = distances.reshape(grid_shape)
    masked_distances = distances[grid_mask.reshape(-1)]
    if masked_distances.size == 0:
        return AnomalyInspectionResult(
            status="retake_photo",
            message="No hay suficientes parches dentro de la mascara. Tome otra foto.",
            matched_reference="anchor",
            anomaly_score=None,
            anomaly_threshold=anomaly_threshold or float(profile["anomaly_threshold"]),
            quality=quality,
            difference_regions=[],
            aligned_image=None,
            heatmap_image=None,
            overlay_image=None,
            valid_mask_ratio=coverage,
        )

    score = float(np.percentile(masked_distances, 99))
    threshold = float(anomaly_threshold if anomaly_threshold is not None else profile["anomaly_threshold"])
    heat_threshold = float(heatmap_threshold if heatmap_threshold is not None else profile["heatmap_threshold"])
    region_area = int(min_region_area if min_region_area is not None else profile["min_region_area"])

    heatmap = _upsample_heatmap(grid_distances, aligned.shape[:2])
    heatmap = _mask_heatmap(heatmap, combined_mask)
    regions, binary_mask = _regions_from_heatmap(heatmap, heat_threshold, region_area)

    status = "review" if score > threshold or regions else "correct"
    message = "Regiones anomalas detectadas; requiere revision." if status == "review" else "La zona coincide con el perfil normal."

    evidence_base = Path(evidence_dir) / family / zone_id / image_path.stem
    evidence_base.mkdir(parents=True, exist_ok=True)
    aligned_path = evidence_base / "aligned.jpg"
    heatmap_path = evidence_base / "heatmap.jpg"
    overlay_path = evidence_base / "overlay.jpg"
    cv2.imwrite(str(aligned_path), aligned)
    cv2.imwrite(str(heatmap_path), _heatmap_image(heatmap, combined_mask))
    cv2.imwrite(str(overlay_path), _overlay_image(aligned, heatmap, binary_mask, regions))

    return AnomalyInspectionResult(
        status=status,
        message=message,
        matched_reference="anchor",
        anomaly_score=score,
        anomaly_threshold=threshold,
        quality=quality,
        difference_regions=regions,
        aligned_image=str(aligned_path),
        heatmap_image=str(heatmap_path),
        overlay_image=str(overlay_path),
        valid_mask_ratio=coverage,
    )


class PatchFeatureExtractor:
    def __init__(self, backend: str, image_size: int, location_weight: float):
        if backend not in {"resnet18", "classical"}:
            raise ValueError("feature backend must be resnet18 or classical")
        self.backend = backend
        self.image_size = int(image_size)
        self.location_weight = float(location_weight)
        self.pretrained = False
        self._torch = None
        self._model = None
        self._device = "cpu"
        if backend == "resnet18":
            self._init_resnet18()

    def extract(self, image, mask) -> tuple[Any, Any, tuple[int, int]]:
        if self.backend == "resnet18":
            visual, grid_shape = self._extract_resnet18(image)
        else:
            visual, grid_shape = self._extract_classical(image)
        visual = _l2_normalize(visual.astype("float32"))
        features = _append_location_features(visual, grid_shape, self.location_weight)
        grid_mask = _grid_mask(mask, grid_shape)
        return features.astype("float32"), grid_mask, grid_shape

    def _init_resnet18(self) -> None:
        try:
            import torch
            from torchvision import models
        except ImportError as exc:
            raise RuntimeError('Install vision dependencies: python3 -m pip install -e ".[vision]"') from exc

        weights = None
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            self.pretrained = True
        except Exception:
            model = models.resnet18(weights=None)
            self.pretrained = False

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        ).to(self._device)
        self._model.eval()

    def _extract_resnet18(self, image) -> tuple[Any, tuple[int, int]]:
        torch = self._torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self._device)
        with torch.no_grad():
            output = self._model(tensor).detach().cpu().numpy()[0]
        channels, height, width = output.shape
        features = output.transpose(1, 2, 0).reshape(height * width, channels)
        return features, (height, width)

    def _extract_classical(self, image) -> tuple[Any, tuple[int, int]]:
        resized = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype("float32") / 255.0
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
        edge = edge / max(float(edge.max()), 1e-6)
        mean = cv2.GaussianBlur(gray, (9, 9), 0)
        std = np.sqrt(cv2.GaussianBlur((gray - mean) ** 2, (9, 9), 0))
        grid = max(16, self.image_size // 8)
        channels = [
            cv2.resize(lab[:, :, channel], (grid, grid), interpolation=cv2.INTER_AREA)
            for channel in range(3)
        ]
        channels.append(cv2.resize(edge, (grid, grid), interpolation=cv2.INTER_AREA))
        channels.append(cv2.resize(std, (grid, grid), interpolation=cv2.INTER_AREA))
        stacked = np.stack(channels, axis=-1)
        return stacked.reshape(grid * grid, stacked.shape[-1]), (grid, grid)


def _nearest_distances(query, memory_bank, chunk_size: int = 1024):
    query = query.astype("float32")
    bank = memory_bank.astype("float32")
    bank_norm = np.sum(bank * bank, axis=1)[None, :]
    out = np.empty(query.shape[0], dtype="float32")
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        chunk_norm = np.sum(chunk * chunk, axis=1)[:, None]
        dist2 = np.maximum(chunk_norm + bank_norm - 2.0 * chunk @ bank.T, 0.0)
        out[start : start + chunk_size] = np.sqrt(np.min(dist2, axis=1))
    return out


def _subsample_bank(memory_bank, max_bank_patches: int):
    if memory_bank.shape[0] <= max_bank_patches:
        return memory_bank
    rng = np.random.default_rng(13)
    indices = rng.choice(memory_bank.shape[0], size=max_bank_patches, replace=False)
    return memory_bank[np.sort(indices)]


def _append_location_features(features, grid_shape: tuple[int, int], location_weight: float):
    height, width = grid_shape
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype="float32"),
        np.linspace(0.0, 1.0, width, dtype="float32"),
        indexing="ij",
    )
    coords = np.stack([ys.reshape(-1), xs.reshape(-1)], axis=1) * location_weight
    return np.concatenate([features, coords], axis=1)


def _grid_mask(mask, grid_shape: tuple[int, int]):
    height, width = grid_shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA) > 127


def _l2_normalize(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-6)


def _read_mask(path: str | Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]


def _resize_mask(mask, shape: tuple[int, int]):
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _combine_masks(base_mask, valid_mask):
    if valid_mask is None:
        return base_mask
    return cv2.bitwise_and(base_mask, _resize_mask(valid_mask, base_mask.shape[:2]))


def _mask_coverage(base_mask, combined_mask) -> float:
    base_count = float(np.count_nonzero(base_mask))
    if base_count == 0:
        return 0.0
    return float(np.count_nonzero(combined_mask)) / base_count


def _upsample_heatmap(grid_distances, shape: tuple[int, int]):
    heatmap = cv2.resize(grid_distances.astype("float32"), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3.0)
    return heatmap


def _mask_heatmap(heatmap, mask):
    masked = heatmap.copy()
    masked[mask <= 0] = 0
    return masked


def _regions_from_heatmap(heatmap, threshold: float, min_region_area: int):
    binary = (heatmap > threshold).astype("uint8") * 255
    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[DifferenceRegion] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_region_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        regions.append(DifferenceRegion(x=x, y=y, width=width, height=height, area=area))
    regions.sort(key=lambda region: region.area, reverse=True)
    return regions, binary


def _heatmap_image(heatmap, mask):
    masked_values = heatmap[mask > 0]
    top = float(np.percentile(masked_values, 99)) if masked_values.size else 1.0
    normalized = np.clip(heatmap / max(top, 1e-6), 0.0, 1.0)
    image = (normalized * 255).astype("uint8")
    image[mask <= 0] = 0
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def _overlay_image(aligned, heatmap, binary_mask, regions: list[DifferenceRegion]):
    heatmap_img = _heatmap_image(heatmap, np.full(aligned.shape[:2], 255, dtype=np.uint8))
    overlay = cv2.addWeighted(aligned, 0.72, heatmap_img, 0.28, 0)
    overlay[binary_mask <= 0] = aligned[binary_mask <= 0]
    for region in regions:
        cv2.rectangle(
            overlay,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            (0, 0, 255),
            3,
        )
    return overlay


def _default_thresholds(feature_backend: str) -> tuple[float, float]:
    if feature_backend == "classical":
        return 0.18, 0.18
    return 0.65, 0.65


def _zone_dir(anomaly_dir: str | Path, family: str, zone_id: str) -> Path:
    return Path(anomaly_dir) / family / zone_id


def _require_deps() -> None:
    if cv2 is None or np is None:
        raise RuntimeError('Install vision dependencies: python3 -m pip install -e ".[vision]"')
