import math

import numpy as np
import pytest

from pt_tem_cv.morphometry import circularity, closed_polygon_perimeter, projected_particle_metrics
from pt_tem_cv.scale import calibrated_nm_per_pixel
from pt_tem_cv.substrate import majority_consensus


def test_majority_consensus_three_masks() -> None:
    masks = [
        np.array([[1, 0], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 0]], dtype=bool),
        np.array([[0, 1], [1, 0]], dtype=bool),
    ]
    assert majority_consensus(masks).tolist() == [[True, True], [True, False]]


def test_square_morphometry() -> None:
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert closed_polygon_perimeter(square) == pytest.approx(8.0)
    assert circularity(4.0, 8.0) == pytest.approx(math.pi / 4.0)
    metrics = projected_particle_metrics(4.0, square, 0.5)
    assert metrics["projected_area_nm2"] == pytest.approx(1.0)
    assert metrics["projected_perimeter_nm"] == pytest.approx(4.0)


def test_verified_scale_conversion() -> None:
    assert calibrated_nm_per_pixel(
        {
            "image_id": "synthetic_001",
            "scale_bar_length_px": 250,
            "scale_bar_value": 50,
            "scale_bar_unit": "nm",
            "verification_status": "verified",
        }
    ) == pytest.approx(0.2)


def test_unverified_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="accepted or verified"):
        calibrated_nm_per_pixel(
            {
                "image_id": "synthetic_001",
                "scale_bar_length_px": 250,
                "scale_bar_value": 50,
                "scale_bar_unit": "nm",
                "verification_status": "pending",
            }
        )
