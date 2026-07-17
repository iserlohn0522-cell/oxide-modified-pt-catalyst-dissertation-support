from __future__ import annotations

from collections.abc import Sequence

Box = tuple[float, float, float, float]
ScoredBox = tuple[Box, float]
ROTATIONS = (0, 90, 180, 270)


def tile_starts(length: int, patch_size: int, stride: int) -> list[int]:
    """Return edge-complete sliding-window start coordinates for one axis."""
    length = int(length)
    patch_size = int(patch_size)
    stride = int(stride)
    if length <= 0:
        raise ValueError("length must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def rotate_point(x: float, y: float, angle: int, width: int, height: int) -> tuple[float, float]:
    if angle == 0:
        return float(x), float(y)
    if angle == 90:
        return float(height - 1 - y), float(x)
    if angle == 180:
        return float(width - 1 - x), float(height - 1 - y)
    if angle == 270:
        return float(y), float(width - 1 - x)
    raise ValueError(f"Unsupported rotation: {angle}")


def rotate_box(box: Box, angle: int, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    corners = (
        rotate_point(x1, y1, angle, width, height),
        rotate_point(x2, y1, angle, width, height),
        rotate_point(x2, y2, angle, width, height),
        rotate_point(x1, y2, angle, width, height),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def box_fully_inside(box: Box, x0: int, y0: int, patch_size: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 >= x0 and y1 >= y0 and x2 <= x0 + patch_size and y2 <= y0 + patch_size


def box_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def non_maximum_suppression(predictions: Sequence[ScoredBox], iou_threshold: float) -> list[ScoredBox]:
    """Merge overlapping tile predictions in source-image coordinates."""
    ordered = sorted(predictions, key=lambda item: item[1], reverse=True)
    kept: list[ScoredBox] = []
    for box, score in ordered:
        if all(box_iou(box, kept_box) < float(iou_threshold) for kept_box, _ in kept):
            kept.append((box, float(score)))
    return kept


__all__ = [
    "Box",
    "ROTATIONS",
    "ScoredBox",
    "box_fully_inside",
    "box_iou",
    "non_maximum_suppression",
    "rotate_box",
    "rotate_point",
    "tile_starts",
]
