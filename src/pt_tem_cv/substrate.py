from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def build_substrate_model() -> Any:
    """Build the U-Net/ResNet-34 architecture used for substrate masks."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Substrate inference requires the 'substrate' optional dependencies") from exc
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)


def load_substrate_state_dict(path: Path, *, device: str = "cpu") -> Any:
    """Load a release checkpoint containing only a model state dictionary."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Substrate inference requires PyTorch") from exc
    state_dict = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Expected a non-empty PyTorch state dictionary")
    return state_dict


def predict_substrate_mask(
    image: Image.Image | np.ndarray,
    model: Any,
    *,
    threshold: float,
    image_size: int = 768,
    device: str = "cpu",
) -> np.ndarray:
    """Return a binary mask in the source image coordinates."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Substrate inference requires PyTorch") from exc

    rgb = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")
    source_size = rgb.size
    resized = rgb.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1))[None].to(device)
    model = model.to(device).eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor)).detach().cpu().numpy()[0, 0]
    small_mask = probability >= float(threshold)
    return np.asarray(
        Image.fromarray(small_mask.astype(np.uint8) * 255, mode="L").resize(source_size, Image.Resampling.NEAREST)
    ) > 0


def majority_consensus(masks: Sequence[np.ndarray], *, required_votes: int | None = None) -> np.ndarray:
    """Combine aligned binary masks by majority vote."""
    if not masks:
        raise ValueError("At least one mask is required")
    boolean_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    shape = boolean_masks[0].shape
    if any(mask.shape != shape for mask in boolean_masks):
        raise ValueError("All masks must have the same shape")
    votes = int(required_votes) if required_votes is not None else len(boolean_masks) // 2 + 1
    if not 1 <= votes <= len(boolean_masks):
        raise ValueError("required_votes must be between one and the number of masks")
    return np.sum(boolean_masks, axis=0) >= votes


__all__ = [
    "build_substrate_model",
    "load_substrate_state_dict",
    "majority_consensus",
    "predict_substrate_mask",
]
