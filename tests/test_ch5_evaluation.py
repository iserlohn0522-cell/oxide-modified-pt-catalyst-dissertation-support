from __future__ import annotations

import csv
import json
from pathlib import Path

from pt_tem_cv.evaluation import evaluate_source_images


def test_merges_overlapping_tile_predictions_before_source_scoring(tmp_path: Path) -> None:
    label_path = tmp_path / "sample.json"
    label_path.write_text(
        json.dumps(
            {
                "imageWidth": 64,
                "imageHeight": 64,
                "shapes": [
                    {"label": "Pt_NPs", "shape_type": "rectangle", "points": [[20, 20], [30, 30]]},
                    {"label": "Pt_NPs", "shape_type": "rectangle", "points": [[40, 40], [50, 50]]},
                    {"label": "ignore_region", "shape_type": "rectangle", "points": [[35, 35], [55, 55]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    image_path = tmp_path / "unused.png"
    image_path.write_bytes(b"not read by source evaluation")
    source_manifest = tmp_path / "source.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "image_path", "label_path"])
        writer.writerow(["sample", image_path.name, label_path.name])
    patch_manifest = tmp_path / "patches.csv"
    with patch_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "split", "rotation_deg", "tile_x0", "tile_y0", "patch_name"],
        )
        writer.writeheader()
        writer.writerow({"image_id": "sample", "split": "val", "rotation_deg": 0, "tile_x0": 0, "tile_y0": 0, "patch_name": "p0"})
        writer.writerow({"image_id": "sample", "split": "val", "rotation_deg": 0, "tile_x0": 16, "tile_y0": 16, "patch_name": "p1"})
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"patch_name": "p0", "predictions": [[20, 20, 30, 30, 0.90]]}),
                json.dumps({"patch_name": "p1", "predictions": [[4, 4, 14, 14, 0.80], [24, 24, 34, 34, 0.85]]}),
            ]
        ),
        encoding="utf-8",
    )
    result = evaluate_source_images(source_manifest, patch_manifest, predictions, splits=("val",))
    assert result["aggregate"]["tp"] == 1
    assert result["aggregate"]["fp"] == 0
    assert result["aggregate"]["fn"] == 0
    assert result["aggregate"]["ignored_ground_truth_count"] == 1
    assert result["aggregate"]["ignored_prediction_count"] == 1
    assert result["aggregate"]["ignore_region_count"] == 1
    assert result["per_image"][0]["predicted_count"] == 1
