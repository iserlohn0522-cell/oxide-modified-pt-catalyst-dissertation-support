from __future__ import annotations

from collections.abc import Mapping
from typing import Any


REQUIRED_SCALE_FIELDS = (
    "image_id",
    "scale_bar_length_px",
    "scale_bar_value",
    "scale_bar_unit",
    "verification_status",
)


def calibrated_nm_per_pixel(record: Mapping[str, Any]) -> float:
    """Validate a human-reviewed scale record and return nanometers per pixel."""
    missing = [field for field in REQUIRED_SCALE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing scale fields: {', '.join(missing)}")
    status = str(record["verification_status"]).strip().lower()
    if status not in {"accepted", "verified"}:
        raise ValueError("Scale record must be accepted or verified before physical-unit conversion")
    length_px = float(record["scale_bar_length_px"])
    value = float(record["scale_bar_value"])
    if length_px <= 0 or value <= 0:
        raise ValueError("Scale-bar length and value must be positive")
    unit = str(record["scale_bar_unit"]).strip().lower().replace("μ", "u").replace("µ", "u")
    factor_to_nm = {"nm": 1.0, "um": 1000.0}.get(unit)
    if factor_to_nm is None:
        raise ValueError(f"Unsupported scale unit: {record['scale_bar_unit']!r}")
    return value * factor_to_nm / length_px


__all__ = ["REQUIRED_SCALE_FIELDS", "calibrated_nm_per_pixel"]
