from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_detector_dataset
from .evaluation import evaluate_source_images
from .training import train_detector


def build_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Build a sliding-window YOLO dataset from user-supplied TEM images and labels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    for path in build_detector_dataset(
        args.config,
        args.source_manifest,
        args.split_manifest,
        output_root=args.output_root,
        materialize_assets=not args.manifest_only,
    ):
        print(path)


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Merge tile predictions and evaluate them in source-image coordinates.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-label", default="Pt_NPs")
    parser.add_argument("--ignore-region-label", action="append", dest="ignore_region_labels")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    result = evaluate_source_images(
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
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train one Ultralytics detector on a materialized TEM tile dataset.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        train_detector(
            args.model,
            args.dataset_yaml,
            args.output_dir,
            image_size=args.image_size,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            resume=args.resume,
        )
    )


__all__ = ["build_dataset_main", "evaluate_main", "train_main"]
