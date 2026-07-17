"""Public utilities for the TEM object-detection workflow used in Chapter 5."""

from .evaluation import evaluate_source_images
from .morphometry import circularity, closed_polygon_perimeter, projected_particle_metrics
from .scale import calibrated_nm_per_pixel
from .sam_prompting import detector_boxes_to_sam_prompts
from .substrate import majority_consensus
from .tiling import ROTATIONS, box_iou, non_maximum_suppression, tile_starts

__all__ = [
    "ROTATIONS",
    "box_iou",
    "calibrated_nm_per_pixel",
    "circularity",
    "closed_polygon_perimeter",
    "detector_boxes_to_sam_prompts",
    "evaluate_source_images",
    "majority_consensus",
    "non_maximum_suppression",
    "projected_particle_metrics",
    "tile_starts",
]

__version__ = "1.0.0"
