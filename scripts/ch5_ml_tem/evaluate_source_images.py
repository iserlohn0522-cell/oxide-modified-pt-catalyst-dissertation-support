from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from pt_tem_cv.evaluation import evaluate_source_images


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Merge tile predictions and evaluate them in source-image coordinates.")
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--patch-manifest", type=Path, required=True)
    result.add_argument("--predictions", type=Path, required=True, help="JSONL with patch_name and local xyxy-score predictions.")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--class-label", default="Pt_NPs")
    result.add_argument("--ignore-region-label", action="append", dest="ignore_region_labels")
    result.add_argument("--split", action="append", dest="splits", help="Split to evaluate; may be repeated.")
    result.add_argument("--confidence", type=float, default=0.0)
    result.add_argument("--nms-iou", type=float, default=0.5)
    result.add_argument("--match-iou", type=float, default=0.5)
    return result


def main() -> None:
    args = parser().parse_args()
    summary = evaluate_source_images(
        args.source_manifest,
        args.patch_manifest,
        args.predictions,
        class_label=args.class_label,
        ignore_region_labels=tuple(args.ignore_region_labels or ("ignore_region",)),
        splits=tuple(args.splits or ("val", "test")),
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        match_iou=args.match_iou,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
