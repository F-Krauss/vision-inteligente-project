import json
from pathlib import Path

from mold_inspection.models import InspectionConfig


def test_load_config_and_class_names(tmp_path: Path):
    config_path = tmp_path / "inspection.json"
    config_path.write_text(
        json.dumps(
            {
                "families": {
                    "fam": {
                        "zones": {
                            "zona_01": {
                                "expected": [
                                    {"id": "a", "class_name": "screw"},
                                    {"id": "b", "class_name": "block"},
                                ]
                            }
                        }
                    }
                }
            }
        )
    )

    config = InspectionConfig.load(config_path)

    assert config.class_names() == ["block", "screw"]
    assert config.zone("fam", "zona_01").id == "zona_01"
