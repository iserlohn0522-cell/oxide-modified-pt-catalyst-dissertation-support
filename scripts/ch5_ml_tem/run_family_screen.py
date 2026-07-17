from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
from pt_tem_cv.screening import expand_family_screen
from pt_tem_cv.training import train_detector


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run or list a configurable detector-family screen.")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--dataset-yaml", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch", type=int, default=8)
    result.add_argument("--device", default="0")
    result.add_argument("--list-only", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    runs = expand_family_screen(args.config)
    if args.list_only:
        print(json.dumps([asdict(run) for run in runs], indent=2))
        return
    for run in runs:
        train_detector(
            run.model,
            args.dataset_yaml,
            args.output_root / run.family,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
