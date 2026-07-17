"""Public utilities for the TEM object-detection workflow used in Chapter 5."""

from .evaluation import evaluate_source_images
from .tiling import ROTATIONS, box_iou, non_maximum_suppression, tile_starts

__all__ = [
    "ROTATIONS",
    "box_iou",
    "evaluate_source_images",
    "non_maximum_suppression",
    "tile_starts",
]

__version__ = "1.0.0"
