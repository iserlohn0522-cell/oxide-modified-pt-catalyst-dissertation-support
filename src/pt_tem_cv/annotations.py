from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

from .tiling import Box


def load_labelme(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def shape_box(shape: Mapping[str, Any]) -> Box | None:
    points = shape.get("points", []) or []
    if not isinstance(points, list) or not points:
        return None
    shape_type = str(shape.get("shape_type") or "polygon").lower()
    if shape_type == "circle" and len(points) >= 2:
        (center_x, center_y), (edge_x, edge_y) = points[:2]
        radius = ((float(edge_x) - float(center_x)) ** 2 + (float(edge_y) - float(center_y)) ** 2) ** 0.5
        return (
            float(center_x) - radius,
            float(center_y) - radius,
            float(center_x) + radius,
            float(center_y) + radius,
        )
    valid_points = [point for point in points if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not valid_points:
        return None
    xs = [float(point[0]) for point in valid_points]
    ys = [float(point[1]) for point in valid_points]
    return min(xs), min(ys), max(xs), max(ys)


def boxes_for_label(data: Mapping[str, Any], class_label: str) -> list[Box]:
    boxes: list[Box] = []
    for shape in data.get("shapes", []) or []:
        if str(shape.get("label") or "") != class_label:
            continue
        box = shape_box(shape)
        if box is not None:
            boxes.append(box)
    return boxes


def regions_for_labels(data: Mapping[str, Any], labels: Sequence[str]) -> list[dict[str, Any]]:
    label_set = {str(label) for label in labels}
    regions: list[dict[str, Any]] = []
    for shape in data.get("shapes", []) or []:
        if str(shape.get("label") or "") not in label_set:
            continue
        box = shape_box(shape)
        if box is None:
            continue
        points = [
            [float(point[0]), float(point[1])]
            for point in shape.get("points", []) or []
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        regions.append(
            {
                "label": str(shape.get("label") or ""),
                "shape_type": str(shape.get("shape_type") or "polygon").lower(),
                "points": points,
                "box": box,
            }
        )
    return regions


def point_in_region(x: float, y: float, region: Mapping[str, Any]) -> bool:
    x1, y1, x2, y2 = [float(value) for value in region["box"]]
    if not (x1 <= x <= x2 and y1 <= y <= y2):
        return False
    shape_type = str(region.get("shape_type") or "polygon").lower()
    points = region.get("points", []) or []
    if shape_type == "circle" and len(points) >= 2:
        center_x, center_y = map(float, points[0][:2])
        edge_x, edge_y = map(float, points[1][:2])
        radius_squared = (edge_x - center_x) ** 2 + (edge_y - center_y) ** 2
        return (float(x) - center_x) ** 2 + (float(y) - center_y) ** 2 <= radius_squared
    if shape_type in {"rectangle", "box"} or len(points) < 3:
        return True
    inside = False
    previous = len(points) - 1
    for index, point in enumerate(points):
        current_x, current_y = map(float, point[:2])
        previous_x, previous_y = map(float, points[previous][:2])
        if (current_y > y) != (previous_y > y):
            crossing_x = (previous_x - current_x) * (float(y) - current_y) / ((previous_y - current_y) or 1e-12) + current_x
            if float(x) < crossing_x:
                inside = not inside
        previous = index
    return inside


def box_center_in_regions(box: Box, regions: Sequence[Mapping[str, Any]]) -> bool:
    x1, y1, x2, y2 = box
    return any(point_in_region((x1 + x2) * 0.5, (y1 + y2) * 0.5, region) for region in regions)


def image_dimensions(data: Mapping[str, Any], source: Path | None = None) -> tuple[int, int]:
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        suffix = f" in {source}" if source else ""
        raise ValueError(f"Missing positive imageWidth/imageHeight{suffix}")
    return width, height


__all__ = [
    "box_center_in_regions",
    "boxes_for_label",
    "image_dimensions",
    "load_labelme",
    "point_in_region",
    "regions_for_labels",
    "shape_box",
]
