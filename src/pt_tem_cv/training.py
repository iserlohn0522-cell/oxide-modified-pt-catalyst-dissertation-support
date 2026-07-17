from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def build_train_overrides(augment_options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the restrained detector augmentations used for TEM image training."""
    options = dict(augment_options or {})
    return {
        "augment": True,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "auto_augment": None,
        "fliplr": 0.5 if bool(options.get("flip", False)) else 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "translate": float(options.get("translate", 0.1)),
        "scale": float(options.get("scale", 0.1)),
        "shear": 0.0,
        "perspective": 0.0,
    }


def train_detector(
    model: str,
    dataset_yaml: Path,
    output_dir: Path,
    *,
    image_size: int = 512,
    epochs: int = 100,
    batch: int = 8,
    device: str | int = 0,
    workers: int = 4,
    seed: int = 0,
    augment_options: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> Path:
    """Train one Ultralytics detector; the optional dependency is imported lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("Detector training requires: pip install 'pt-tem-cv[ml]'") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_weight = output_dir / "train" / "weights" / "last.pt"
    if resume and last_weight.exists():
        detector = YOLO(str(last_weight))
        detector.train(resume=True)
    else:
        detector = YOLO(str(model))
        detector.train(
            data=str(Path(dataset_yaml)),
            imgsz=int(image_size),
            epochs=int(epochs),
            batch=int(batch),
            device=device,
            workers=int(workers),
            seed=int(seed),
            project=str(output_dir),
            name="train",
            exist_ok=True,
            **build_train_overrides(augment_options),
        )
    return output_dir / "train" / "weights" / "best.pt"


__all__ = ["build_train_overrides", "train_detector"]
