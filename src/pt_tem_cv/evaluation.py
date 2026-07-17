from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .annotations import box_center_in_regions, boxes_for_label, load_labelme, regions_for_labels
from .dataset import load_source_manifest
from .tiling import Box, ScoredBox, box_iou, non_maximum_suppression


def _read_patch_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_prediction_dump(path: Path) -> dict[str, list[ScoredBox]]:
    by_patch: dict[str, list[ScoredBox]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            patch_name = str(record["patch_name"])
            predictions: list[ScoredBox] = []
            for item in record.get("predictions", []) or []:
                if isinstance(item, Mapping):
                    values = [*item["box"], item["score"]]
                else:
                    values = list(item)
                if len(values) < 5:
                    raise ValueError(f"Prediction on line {line_number} has fewer than five values")
                x1, y1, x2, y2, score = map(float, values[:5])
                predictions.append(((x1, y1, x2, y2), score))
            by_patch[patch_name] = predictions
    return by_patch


def _match_counts(gt_boxes: Sequence[Box], predictions: Sequence[ScoredBox], iou_threshold: float) -> dict[str, float | int]:
    unmatched = set(range(len(gt_boxes)))
    true_positive = 0
    for prediction, _score in sorted(predictions, key=lambda item: item[1], reverse=True):
        best_index = -1
        best_iou = 0.0
        for index in unmatched:
            overlap = box_iou(prediction, gt_boxes[index])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= float(iou_threshold):
            unmatched.remove(best_index)
            true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(gt_boxes) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_source_images(
    source_manifest_path: Path,
    patch_manifest_path: Path,
    prediction_dump_path: Path,
    *,
    class_label: str = "Pt_NPs",
    ignore_region_labels: Sequence[str] = ("ignore_region",),
    splits: Sequence[str] = ("val", "test"),
    confidence: float = 0.0,
    nms_iou: float = 0.5,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Merge rotation-zero tile predictions and score them on source images."""
    sources = load_source_manifest(Path(source_manifest_path))
    split_set = set(splits)
    patch_rows = [
        row
        for row in _read_patch_rows(Path(patch_manifest_path))
        if str(row.get("split", "")) in split_set and int(float(row.get("rotation_deg", 0))) == 0
    ]
    prediction_dump = _load_prediction_dump(Path(prediction_dump_path))
    predictions_by_image: dict[str, list[ScoredBox]] = {}
    evaluated_image_ids: set[str] = set()
    for row in patch_rows:
        image_id = str(row["image_id"])
        evaluated_image_ids.add(image_id)
        x0 = float(row["tile_x0"])
        y0 = float(row["tile_y0"])
        for box, score in prediction_dump.get(str(row["patch_name"]), []):
            if score < float(confidence):
                continue
            x1, y1, x2, y2 = box
            predictions_by_image.setdefault(image_id, []).append(((x1 + x0, y1 + y0, x2 + x0, y2 + y0), score))

    per_image: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    total_ignored_gt = total_ignored_predictions = total_ignore_regions = 0
    count_errors: list[float] = []
    for image_id in sorted(evaluated_image_ids):
        if image_id not in sources:
            raise KeyError(f"{image_id!r} from the patch manifest is absent from the source manifest")
        annotation = load_labelme(sources[image_id].label_path)
        ignore_regions = regions_for_labels(annotation, ignore_region_labels)
        raw_gt_boxes = boxes_for_label(annotation, class_label)
        ignored_gt_boxes = [box for box in raw_gt_boxes if box_center_in_regions(box, ignore_regions)]
        gt_boxes = [box for box in raw_gt_boxes if not box_center_in_regions(box, ignore_regions)]
        raw_merged = non_maximum_suppression(predictions_by_image.get(image_id, []), nms_iou)
        ignored_predictions = [item for item in raw_merged if box_center_in_regions(item[0], ignore_regions)]
        merged = [item for item in raw_merged if not box_center_in_regions(item[0], ignore_regions)]
        metrics = _match_counts(gt_boxes, merged, match_iou)
        total_tp += int(metrics["tp"])
        total_fp += int(metrics["fp"])
        total_fn += int(metrics["fn"])
        total_ignored_gt += len(ignored_gt_boxes)
        total_ignored_predictions += len(ignored_predictions)
        total_ignore_regions += len(ignore_regions)
        count_error = len(merged) - len(gt_boxes)
        count_errors.append(float(count_error))
        per_image.append(
            {
                "image_id": image_id,
                "ground_truth_count": len(gt_boxes),
                "predicted_count": len(merged),
                "count_error": count_error,
                "raw_ground_truth_count": len(raw_gt_boxes),
                "ignored_ground_truth_count": len(ignored_gt_boxes),
                "raw_prediction_count": len(raw_merged),
                "ignored_prediction_count": len(ignored_predictions),
                "ignore_region_count": len(ignore_regions),
                "metrics": metrics,
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "scope": "source_images_from_rotation_zero_tiles",
        "splits": list(splits),
        "thresholds": {
            "confidence": float(confidence),
            "nms_iou": float(nms_iou),
            "match_iou": float(match_iou),
            "ignore_region_labels": list(ignore_region_labels),
            "ignore_region_policy": "exclude GT and predictions whose box center lies inside an annotated region",
        },
        "aggregate": {
            "image_count": len(per_image),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ignored_ground_truth_count": total_ignored_gt,
            "ignored_prediction_count": total_ignored_predictions,
            "ignore_region_count": total_ignore_regions,
            "mean_count_bias": statistics.fmean(count_errors) if count_errors else 0.0,
            "mean_absolute_count_error": statistics.fmean(abs(value) for value in count_errors) if count_errors else 0.0,
        },
        "per_image": per_image,
    }


__all__ = ["evaluate_source_images"]
