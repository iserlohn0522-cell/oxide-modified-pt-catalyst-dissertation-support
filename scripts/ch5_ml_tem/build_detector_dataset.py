from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from pt_tem_cv.dataset import build_detector_dataset


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build a sliding-window YOLO dataset from user-supplied TEM images and labels.")
    result.add_argument("--config", type=Path, required=True, help="Dataset YAML configuration.")
    result.add_argument("--source-manifest", type=Path, required=True, help="CSV listing image and LabelMe JSON paths.")
    result.add_argument("--split-manifest", type=Path, required=True, help="YAML defining train/validation folds.")
    result.add_argument("--output-root", type=Path, default=None, help="Override the output_root in the config.")
    result.add_argument("--manifest-only", action="store_true", help="Write patch rows without image/label assets.")
    return result


def main() -> None:
    args = parser().parse_args()
    written = build_detector_dataset(
        args.config,
        args.source_manifest,
        args.split_manifest,
        output_root=args.output_root,
        materialize_assets=not args.manifest_only,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
