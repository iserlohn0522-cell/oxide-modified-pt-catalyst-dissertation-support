from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def closed_polygon_perimeter(points_xy: Sequence[Sequence[float]]) -> float:
    """Return the Euclidean perimeter of a closed polygon."""
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("A polygon requires at least three x-y points")
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).sum())


def circularity(area: float, perimeter: float, *, clip: bool = True) -> float:
    """Calculate 4*pi*area/perimeter^2 for a projected particle outline."""
    if area < 0:
        raise ValueError("area must be non-negative")
    if perimeter <= 0:
        raise ValueError("perimeter must be positive")
    value = 4.0 * math.pi * float(area) / float(perimeter) ** 2
    return min(1.0, max(0.0, value)) if clip else value


def projected_particle_metrics(area_px2: float, contour_xy: Sequence[Sequence[float]], nm_per_px: float) -> dict[str, float]:
    """Convert one two-dimensional particle outline to calibrated metrics."""
    if nm_per_px <= 0:
        raise ValueError("nm_per_px must be positive")
    perimeter_px = closed_polygon_perimeter(contour_xy)
    return {
        "projected_area_nm2": float(area_px2) * float(nm_per_px) ** 2,
        "projected_perimeter_nm": perimeter_px * float(nm_per_px),
        "equivalent_diameter_nm": 2.0 * math.sqrt(float(area_px2) * float(nm_per_px) ** 2 / math.pi),
        "circularity": circularity(float(area_px2), perimeter_px),
    }


__all__ = ["circularity", "closed_polygon_perimeter", "projected_particle_metrics"]
