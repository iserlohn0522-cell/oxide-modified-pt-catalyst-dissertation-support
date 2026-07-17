from __future__ import annotations

import numpy as np

from pt_tem_cv.augment import apply_train_transforms, sample_transform_parameters


def test_range_sampling_is_deterministic_per_sample() -> None:
    options = {"seed": 17, "contrast_amplitude": 0.3, "sharpness_amplitude": 0.5, "remap_tail_clip": 2.0}
    assert sample_transform_parameters(options, sample_key="tile-a") == sample_transform_parameters(options, sample_key="tile-a")
    assert sample_transform_parameters(options, sample_key="tile-a") != sample_transform_parameters(options, sample_key="tile-b")


def test_transform_preserves_shape_and_dtype() -> None:
    image = np.repeat(np.arange(64, dtype=np.uint8).reshape(8, 8, 1), 3, axis=2)
    transformed = apply_train_transforms(
        image,
        {"seed": 3, "grayscale": True, "contrast_amplitude": 0.2, "sharpness_amplitude": 0.4},
        sample_key="tile",
    )
    assert transformed.shape == image.shape
    assert transformed.dtype == np.uint8
    assert np.array_equal(transformed[:, :, 0], transformed[:, :, 1])
