from pathlib import Path

from mold_inspection.dataset import audit_split, split_manifest, write_manifest


def test_split_keeps_session_group_in_one_split(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "split.csv"
    rows = []
    for mold_index in range(10):
        for image_index in range(3):
            rows.append(
                {
                    "image_path": f"image_{mold_index}_{image_index}.jpg",
                    "family": "fam",
                    "mold_id": f"molde_{mold_index}",
                    "session_id": "sesion_1",
                    "zone_id": "zona_01",
                    "state": "correct",
                    "source_path": "",
                    "captured_at": "",
                    "width": "",
                    "height": "",
                }
            )
    write_manifest(manifest, rows)

    split_rows = split_manifest(manifest, output, val_ratio=0.2, test_ratio=0.2, seed=1)

    assert not audit_split(split_rows)
