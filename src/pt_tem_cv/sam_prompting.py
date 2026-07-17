from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def detector_boxes_to_sam_prompts(boxes_xyxy: Sequence[Sequence[float]], *, image_width: int, image_height: int) -> np.ndarray:
    """Validate source-coordinate detector boxes for a box-prompted mask model."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (n, 4)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if np.any(~np.isfinite(boxes)):
        raise ValueError("boxes must contain finite values")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("each box must have positive width and height")
    if np.any(boxes[:, 0] < 0) or np.any(boxes[:, 1] < 0):
        raise ValueError("box coordinates must not be negative")
    if np.any(boxes[:, 2] > image_width) or np.any(boxes[:, 3] > image_height):
        raise ValueError("boxes must lie within the source image")
    return boxes


__all__ = ["detector_boxes_to_sam_prompts"]
