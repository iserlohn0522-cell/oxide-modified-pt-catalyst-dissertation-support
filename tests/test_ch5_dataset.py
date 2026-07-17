from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from pt_tem_cv.dataset import build_detector_dataset


def _write_record(root: Path, image_id: str, x_offset: int = 0, add_ignore_region: bool = False) -> tuple[Path, Path]:
    image_path = root / "images" / f"{image_id}.png"
    label_path = root / "labels" / f"{image_id}.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2), mode="RGB").save(image_path)
    shapes = [
        {
            "label": "Pt_NPs",
            "shape_type": "rectangle",
            "points": [[8 + x_offset, 8], [16 + x_offset, 16]],
        }
    ]
    if add_ignore_region:
        shapes.append(
            {
                "label": "ignore_region",
                "shape_type": "rectangle",
                "points": [[32, 32], [63, 63]],
            }
        )
    label_path.write_text(
        json.dumps(
            {
                "imageWidth": 64,
                "imageHeight": 64,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )
    return image_path, label_path


def test_builds_portable_sliding_window_dataset(tmp_path: Path) -> None:
    records = []
    for image_id, offset in (("sample_001", 0), ("sample_002", 16), ("sample_003", 32)):
        image_path, label_path = _write_record(tmp_path, image_id, offset, add_ignore_region=image_id == "sample_001")
        records.append((image_id, image_path, label_path))
    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "image_path", "label_path"])
        for image_id, image_path, label_path in records:
            writer.writerow([image_id, image_path.relative_to(tmp_path), label_path.relative_to(tmp_path)])
    split_manifest = tmp_path / "splits.yaml"
    split_manifest.write_text(
        yaml.safe_dump(
            {"folds": {"fold_1": {"train": ["sample_001"], "val": ["sample_002"], "test": ["sample_003"]}}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "patch_size": 32,
                "stride": 16,
                "rotations": [0, 90],
                "class_label": "Pt_NPs",
                "include_empty": True,
                "ignore_region_labels": ["ignore_region"],
                "exclude_ignore_region_patches": True,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "built"
    manifests = build_detector_dataset(config, source_manifest, split_manifest, output_root=output)
    assert manifests == [output / "fold_1" / "patch_manifest.csv"]
    with manifests[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 46
    assert {row["rotation_deg"] for row in rows} == {"0", "90"}
    assert all(not Path(row["patch_image"]).is_absolute() for row in rows)
    first_image = output / "fold_1" / rows[0]["patch_image"]
    with Image.open(first_image) as image:
        assert image.size == (32, 32)
    dataset_yaml = yaml.safe_load((output / "fold_1" / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["test"] == "images/test"
    with (output / "fold_1" / "excluded_patch_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        excluded_rows = list(csv.DictReader(handle))
    assert len(excluded_rows) == 8
    assert all(row["excluded_by_ignore_region"] == "1" for row in excluded_rows)
