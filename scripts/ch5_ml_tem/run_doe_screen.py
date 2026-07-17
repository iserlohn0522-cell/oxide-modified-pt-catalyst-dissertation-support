from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
from pt_tem_cv.dataset import build_detector_dataset
from pt_tem_cv.screening import expand_doe_screen
from pt_tem_cv.training import train_detector


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run or list a factorial image-augmentation DOE screen.")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--dataset-config", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--split-manifest", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--fold", default="fold_1")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch", type=int, default=8)
    result.add_argument("--device", default="0")
    result.add_argument("--list-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    runs = expand_doe_screen(args.config)
    if args.list_only:
        print(json.dumps([asdict(run) for run in runs], indent=2))
        return
    for index, run in enumerate(runs):
        run_dir = args.output_root / run.family / f"{index:02d}_{run.augment_name}"
        build_detector_dataset(
            args.dataset_config,
            args.source_manifest,
            args.split_manifest,
            output_root=run_dir / "dataset",
            train_transforms_override=run.train_transforms,
        )
        dataset_yaml = run_dir / "dataset" / args.fold / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Requested fold was not built: {dataset_yaml}")
        train_detector(
            run.model,
            dataset_yaml,
            run_dir / "model",
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            augment_options=run.train_transforms,
        )


if __name__ == "__main__":
    main()
