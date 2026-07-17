from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from pt_tem_cv.training import train_detector


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train one Ultralytics detector on a materialized TEM tile dataset.")
    result.add_argument("--model", required=True, help="Ultralytics model name or user-supplied checkpoint path.")
    result.add_argument("--dataset-yaml", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--image-size", type=int, default=512)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch", type=int, default=8)
    result.add_argument("--device", default="0")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--resume", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    best = train_detector(
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
    print(best)


if __name__ == "__main__":
    main()
