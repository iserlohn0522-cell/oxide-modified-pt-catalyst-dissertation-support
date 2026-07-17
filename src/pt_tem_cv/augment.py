from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32), 0, 255).astype(np.uint8)


def _stable_rng(seed: int, sample_key: str) -> np.random.Generator:
    token = f"{int(seed)}::{sample_key}".encode("utf-8")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder="big", signed=False))


def sample_transform_parameters(
    options: Mapping[str, Any] | None = None,
    *,
    sample_key: str = "",
) -> dict[str, float]:
    options = dict(options or {})
    rng = _stable_rng(int(options.get("seed", 0)), sample_key)

    def sampled_factor(name: str, amplitude_name: str) -> float:
        explicit = options.get(name)
        if explicit is not None:
            return float(explicit)
        lower = options.get(f"{name}_min")
        upper = options.get(f"{name}_max")
        if lower is not None or upper is not None:
            low = float(lower if lower is not None else 1.0)
            high = float(upper if upper is not None else low)
            if high < low:
                raise ValueError(f"{name}_max must be >= {name}_min")
            return float(rng.uniform(low, high))
        amplitude = max(0.0, float(options.get(amplitude_name, 0.0) or 0.0))
        return float(rng.uniform(max(0.0, 1.0 - amplitude), 1.0 + amplitude)) if amplitude else 1.0

    sharpness_amplitude = max(0.0, float(options.get("sharpness_amplitude", 0.0) or 0.0))
    tail_clip = max(0.0, float(options.get("remap_tail_clip", 0.0) or 0.0))
    tail = float(rng.uniform(0.5, tail_clip)) if tail_clip > 0.5 else (0.5 if tail_clip > 0.0 else 0.0)
    return {
        "contrast_factor": sampled_factor("contrast_factor", "contrast_amplitude"),
        "brightness_factor": sampled_factor("brightness_factor", "brightness_amplitude"),
        "sharpness_factor": float(rng.uniform(1.0, 1.0 + sharpness_amplitude)) if sharpness_amplitude else 1.0,
        "remap_low_percentile": tail,
        "remap_high_percentile": 100.0 - tail if tail else 100.0,
    }


def apply_train_transforms(
    image: np.ndarray,
    options: Mapping[str, Any] | None = None,
    *,
    sample_key: str = "",
) -> np.ndarray:
    options = dict(options or {})
    sampled = sample_transform_parameters(options, sample_key=sample_key)
    result = Image.fromarray(_to_u8(image), mode="RGB")
    if bool(options.get("grayscale", False)):
        result = ImageOps.grayscale(result).convert("RGB")
    if sampled["brightness_factor"] != 1.0:
        result = ImageEnhance.Brightness(result).enhance(sampled["brightness_factor"])
    if bool(options.get("intensity_remap", False)) or sampled["remap_low_percentile"] > 0.0:
        gray = np.asarray(ImageOps.grayscale(result), dtype=np.float32)
        low_percentile = sampled["remap_low_percentile"] or 1.0
        high_percentile = sampled["remap_high_percentile"] if sampled["remap_low_percentile"] else 99.0
        low = float(np.percentile(gray, low_percentile))
        high = float(np.percentile(gray, high_percentile))
        if high > low:
            remapped = np.clip((gray - low) * (255.0 / (high - low)), 0.0, 255.0).astype(np.uint8)
            result = Image.fromarray(np.repeat(remapped[:, :, None], 3, axis=2), mode="RGB")
    if bool(options.get("contrast", False)) and sampled["contrast_factor"] == 1.0:
        sampled["contrast_factor"] = 1.35
    if sampled["contrast_factor"] != 1.0:
        result = ImageEnhance.Contrast(result).enhance(sampled["contrast_factor"])
    if bool(options.get("sharpness", False)) and sampled["sharpness_factor"] == 1.0:
        sampled["sharpness_factor"] = 1.8
    if sampled["sharpness_factor"] != 1.0:
        result = ImageEnhance.Sharpness(result).enhance(sampled["sharpness_factor"])
    return np.asarray(result, dtype=np.uint8)


__all__ = ["apply_train_transforms", "sample_transform_parameters"]
