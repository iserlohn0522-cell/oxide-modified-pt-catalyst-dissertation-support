from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from .annotations import boxes_for_label, load_labelme
from .augment import apply_train_transforms
from .tiling import ROTATIONS, Box, box_fully_inside, rotate_box, tile_starts

PATCH_MANIFEST_COLUMNS = (
    "image_id",
    "split",
    "rotation_deg",
    "tile_x0",
    "tile_y0",
    "patch_size",
    "stride",
    "patch_name",
    "patch_image",
    "patch_label",
    "labels",
    "empty_patch",
    "ignore_region_overlap",
    "excluded_by_ignore_region",
)


@dataclass(frozen=True)
class SourceRecord:
    image_id: str
    image_path: Path
    label_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def _resolve_from(base: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_source_manifest(path: Path) -> dict[str, SourceRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "image_path", "label_path"}
    if not rows:
        raise ValueError(f"Source manifest is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    records: dict[str, SourceRecord] = {}
    for row in rows:
        image_id = str(row["image_id"]).strip()
        if not image_id:
            raise ValueError("image_id must not be empty")
        if image_id in records:
            raise ValueError(f"Duplicate image_id in source manifest: {image_id}")
        records[image_id] = SourceRecord(
            image_id=image_id,
            image_path=_resolve_from(path.parent, row["image_path"]),
            label_path=_resolve_from(path.parent, row["label_path"]),
        )
    return records


def load_split_manifest(path: Path) -> dict[str, dict[str, list[str]]]:
    raw = _load_yaml(Path(path))
    folds = raw.get("folds")
    if not isinstance(folds, dict) or not folds:
        raise ValueError("Split manifest must define a non-empty 'folds' mapping")
    normalized: dict[str, dict[str, list[str]]] = {}
    for fold_name, split_map in folds.items():
        if not isinstance(split_map, dict):
            raise TypeError(f"Fold {fold_name!r} must map split names to image IDs")
        normalized[str(fold_name)] = {
            str(split_name): [str(image_id) for image_id in image_ids]
            for split_name, image_ids in split_map.items()
        }
        train_ids = set(normalized[str(fold_name)].get("train", []))
        val_ids = set(normalized[str(fold_name)].get("val", []))
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(f"Fold {fold_name!r} has train/val overlap: {sorted(overlap)}")
        if not train_ids or not val_ids:
            raise ValueError(f"Fold {fold_name!r} must contain non-empty train and val splits")
    return normalized


def _rotated_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image.copy()
    if angle == 90:
        return np.rot90(image, k=3).copy()
    if angle == 180:
        return np.rot90(image, k=2).copy()
    if angle == 270:
        return np.rot90(image, k=1).copy()
    raise ValueError(f"Unsupported rotation: {angle}")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8)


def _padded_patch(image: np.ndarray, x0: int, y0: int, patch_size: int) -> np.ndarray:
    patch = image[y0 : y0 + patch_size, x0 : x0 + patch_size]
    pad_height = patch_size - patch.shape[0]
    pad_width = patch_size - patch.shape[1]
    if pad_height <= 0 and pad_width <= 0:
        return patch.copy()
    return np.pad(
        patch,
        ((0, max(0, pad_height)), (0, max(0, pad_width)), (0, 0)),
        mode="edge",
    )


def _yolo_box_line(class_id: int, box: Box, x0: int, y0: int, patch_size: int) -> str:
    x1, y1, x2, y2 = box
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    center_x = ((x1 + x2) * 0.5 - x0) / patch_size
    center_y = ((y1 + y2) * 0.5 - y0) / patch_size
    return (
        f"{int(class_id)} {center_x:.6f} {center_y:.6f} "
        f"{width / patch_size:.6f} {height / patch_size:.6f}"
    )


def _boxes_overlap(box_a: Box, box_b: Box) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PATCH_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset_yaml(path: Path, fold_dir: Path, split_names: set[str], class_label: str) -> None:
    payload: dict[str, Any] = {
        "path": fold_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_label},
    }
    if "test" in split_names:
        payload["test"] = "images/test"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _materialize_record(
    record: SourceRecord,
    split_name: str,
    fold_dir: Path,
    *,
    patch_size: int,
    stride: int,
    rotations: Sequence[int],
    class_label: str,
    class_id: int,
    include_empty: bool,
    ignore_region_labels: Sequence[str],
    exclude_ignore_region_patches: bool,
    train_transforms: Mapping[str, Any] | None,
    materialize_assets: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotation = load_labelme(record.label_path)
    source_image = _load_rgb(record.image_path)
    image_height, image_width = source_image.shape[:2]
    annotated_width = int(annotation.get("imageWidth") or image_width)
    annotated_height = int(annotation.get("imageHeight") or image_height)
    if (annotated_width, annotated_height) != (image_width, image_height):
        raise ValueError(f"Image/annotation dimensions differ for {record.image_id}")
    boxes = boxes_for_label(annotation, class_label)
    ignore_boxes = [
        box
        for label in ignore_region_labels
        for box in boxes_for_label(annotation, str(label))
    ]
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for angle in rotations:
        rotated_image = _rotated_image(source_image, int(angle))
        rotated_boxes = [rotate_box(box, int(angle), image_width, image_height) for box in boxes]
        rotated_ignore_boxes = [rotate_box(box, int(angle), image_width, image_height) for box in ignore_boxes]
        rotated_height, rotated_width = rotated_image.shape[:2]
        for y0 in tile_starts(rotated_height, patch_size, stride):
            for x0 in tile_starts(rotated_width, patch_size, stride):
                local_boxes = [box for box in rotated_boxes if box_fully_inside(box, x0, y0, patch_size)]
                patch_name = f"{record.image_id}__r{int(angle):03d}__x{x0:05d}_y{y0:05d}"
                patch_box = (float(x0), float(y0), float(x0 + patch_size), float(y0 + patch_size))
                ignore_overlap = any(_boxes_overlap(patch_box, region) for region in rotated_ignore_boxes)
                base_row = {
                    "image_id": record.image_id,
                    "split": split_name,
                    "rotation_deg": int(angle),
                    "tile_x0": x0,
                    "tile_y0": y0,
                    "patch_size": patch_size,
                    "stride": stride,
                    "patch_name": patch_name,
                    "patch_image": "",
                    "patch_label": "",
                    "labels": len(local_boxes),
                    "empty_patch": int(not local_boxes),
                    "ignore_region_overlap": int(ignore_overlap),
                    "excluded_by_ignore_region": int(ignore_overlap and exclude_ignore_region_patches),
                }
                if ignore_overlap and exclude_ignore_region_patches:
                    excluded_rows.append(base_row)
                    continue
                if not local_boxes and not include_empty:
                    continue
                image_rel = Path("images") / split_name / f"{patch_name}.png"
                label_rel = Path("labels") / split_name / f"{patch_name}.txt"
                if materialize_assets:
                    patch_image = _padded_patch(rotated_image, x0, y0, patch_size)
                    if split_name == "train" and train_transforms:
                        patch_image = apply_train_transforms(patch_image, train_transforms, sample_key=patch_name)
                    image_path = fold_dir / image_rel
                    label_path = fold_dir / label_rel
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(patch_image, mode="RGB").save(image_path)
                    label_path.write_text(
                        "\n".join(_yolo_box_line(class_id, box, x0, y0, patch_size) for box in local_boxes),
                        encoding="utf-8",
                    )
                rows.append(
                    {
                        **base_row,
                        "patch_image": image_rel.as_posix() if materialize_assets else "",
                        "patch_label": label_rel.as_posix() if materialize_assets else "",
                    }
                )
    return rows, excluded_rows


def build_detector_dataset(
    config_path: Path,
    source_manifest_path: Path,
    split_manifest_path: Path,
    *,
    output_root: Path | None = None,
    materialize_assets: bool = True,
    train_transforms_override: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Build portable sliding-window YOLO datasets from user-supplied records."""
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    records = load_source_manifest(Path(source_manifest_path))
    folds = load_split_manifest(Path(split_manifest_path))
    configured_output = Path(config.get("output_root", "outputs/ch5_ml_tem"))
    if output_root is None:
        output_root = configured_output if configured_output.is_absolute() else (Path.cwd() / configured_output)
    output_root = Path(output_root)
    patch_size = int(config.get("patch_size", 512))
    stride = int(config.get("stride", patch_size // 2))
    rotations = tuple(int(value) for value in config.get("rotations", ROTATIONS))
    unsupported = set(rotations) - set(ROTATIONS)
    if unsupported:
        raise ValueError(f"Unsupported rotations: {sorted(unsupported)}")
    class_label = str(config.get("class_label", "Pt_NPs"))
    class_id = int(config.get("class_id", 0))
    include_empty = bool(config.get("include_empty", True))
    ignore_region_labels = tuple(str(value) for value in config.get("ignore_region_labels", ["ignore_region"]))
    exclude_ignore_region_patches = bool(config.get("exclude_ignore_region_patches", False))
    train_transforms = dict(
        train_transforms_override
        if train_transforms_override is not None
        else (config.get("train_transforms", {}) or {})
    )

    written: list[Path] = []
    for fold_name, split_map in folds.items():
        fold_dir = output_root / fold_name
        all_rows: list[dict[str, Any]] = []
        all_excluded_rows: list[dict[str, Any]] = []
        seen_within_fold: dict[str, str] = {}
        for split_name, image_ids in split_map.items():
            for image_id in image_ids:
                if image_id not in records:
                    raise KeyError(f"{image_id!r} is absent from the source manifest")
                previous_split = seen_within_fold.get(image_id)
                if previous_split is not None and previous_split != split_name:
                    raise ValueError(f"{image_id!r} appears in both {previous_split!r} and {split_name!r}")
                seen_within_fold[image_id] = split_name
                record = records[image_id]
                if not record.image_path.exists():
                    raise FileNotFoundError(record.image_path)
                if not record.label_path.exists():
                    raise FileNotFoundError(record.label_path)
                record_rows, excluded_rows = _materialize_record(
                    record,
                    split_name,
                    fold_dir,
                    patch_size=patch_size,
                    stride=stride,
                    rotations=rotations,
                    class_label=class_label,
                    class_id=class_id,
                    include_empty=include_empty,
                    ignore_region_labels=ignore_region_labels,
                    exclude_ignore_region_patches=exclude_ignore_region_patches,
                    train_transforms=train_transforms,
                    materialize_assets=materialize_assets,
                )
                all_rows.extend(record_rows)
                all_excluded_rows.extend(excluded_rows)
        manifest_path = fold_dir / "patch_manifest.csv"
        _write_rows(manifest_path, all_rows)
        _write_rows(fold_dir / "excluded_patch_manifest.csv", all_excluded_rows)
        if materialize_assets:
            _write_dataset_yaml(fold_dir / "dataset.yaml", fold_dir, set(split_map), class_label)
        written.append(manifest_path)
    return written


__all__ = [
    "PATCH_MANIFEST_COLUMNS",
    "SourceRecord",
    "build_detector_dataset",
    "load_source_manifest",
    "load_split_manifest",
]
