from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import json


class ImageState(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SIMULATED_FAULT = "simulated_fault"


class ElementStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    DOUBTFUL = "doubtful"


class ZoneStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    REVIEW = "review"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, float, float, float]) -> "Box":
        if len(values) != 4:
            raise ValueError("bbox/roi must have four values")
        x1, y1, x2, y2 = (float(v) for v in values)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox/roi must be [x1, y1, x2, y2]")
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def intersection_area(self, other: "Box") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def overlap_ratio(self, other: "Box") -> float:
        if self.area == 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class ExpectedElement:
    id: str
    class_name: str
    roi: Box | None = None
    min_confidence: float = 0.55
    min_overlap: float = 0.25
    critical: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedElement":
        return cls(
            id=str(data["id"]),
            class_name=str(data["class_name"]),
            roi=Box.from_list(data["roi"]) if data.get("roi") is not None else None,
            min_confidence=float(data.get("min_confidence", 0.55)),
            min_overlap=float(data.get("min_overlap", 0.25)),
            critical=bool(data.get("critical", True)),
        )


@dataclass(frozen=True)
class ZoneConfig:
    id: str
    description: str = ""
    expected: list[ExpectedElement] = field(default_factory=list)

    @classmethod
    def from_dict(cls, zone_id: str, data: dict[str, Any]) -> "ZoneConfig":
        return cls(
            id=zone_id,
            description=str(data.get("description", "")),
            expected=[ExpectedElement.from_dict(item) for item in data.get("expected", [])],
        )


@dataclass(frozen=True)
class FamilyConfig:
    id: str
    zones: dict[str, ZoneConfig]

    @classmethod
    def from_dict(cls, family_id: str, data: dict[str, Any]) -> "FamilyConfig":
        zones = {
            zone_id: ZoneConfig.from_dict(zone_id, zone_data)
            for zone_id, zone_data in data.get("zones", {}).items()
        }
        return cls(id=family_id, zones=zones)


@dataclass(frozen=True)
class InspectionConfig:
    families: dict[str, FamilyConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InspectionConfig":
        families = {
            family_id: FamilyConfig.from_dict(family_id, family_data)
            for family_id, family_data in data.get("families", {}).items()
        }
        return cls(families=families)

    @classmethod
    def load(cls, path: str | Path) -> "InspectionConfig":
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("Install pyyaml or use JSON config files.") from exc
            data = yaml.safe_load(path.read_text()) or {}
        else:
            data = json.loads(path.read_text())
        return cls.from_dict(data)

    def zone(self, family_id: str, zone_id: str) -> ZoneConfig:
        try:
            return self.families[family_id].zones[zone_id]
        except KeyError as exc:
            raise KeyError(f"Unknown family/zone: {family_id}/{zone_id}") from exc

    def class_names(self) -> list[str]:
        names: set[str] = set()
        for family in self.families.values():
            for zone in family.zones.values():
                for element in zone.expected:
                    names.add(element.class_name)
        return sorted(names)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: Box | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        bbox = data.get("bbox")
        return cls(
            class_name=str(data["class_name"]),
            confidence=float(data.get("confidence", 1.0)),
            bbox=Box.from_list(bbox) if bbox is not None else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.as_list() if self.bbox else None,
        }


@dataclass(frozen=True)
class ElementFinding:
    element_id: str
    class_name: str
    status: ElementStatus
    critical: bool
    confidence: float | None = None
    bbox: Box | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "class_name": self.class_name,
            "status": self.status.value,
            "critical": self.critical,
            "confidence": self.confidence,
            "bbox": self.bbox.as_list() if self.bbox else None,
        }
