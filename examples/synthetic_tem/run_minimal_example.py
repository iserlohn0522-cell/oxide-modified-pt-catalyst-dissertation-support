"""Exercise the public interfaces with generated data; this is not an accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pt_tem_cv.morphometry import projected_particle_metrics
from pt_tem_cv.scale import calibrated_nm_per_pixel
from pt_tem_cv.substrate import majority_consensus
from pt_tem_cv.tiling import non_maximum_suppression, tile_starts


def run_example() -> dict[str, object]:
    width = height = 1024
    starts = tile_starts(width, patch_size=512, stride=384)
    simulated_tile_predictions = [
        ((202.0, 202.0, 302.0, 302.0), 0.92),
        ((204.0, 203.0, 301.0, 303.0), 0.88),
        ((700.0, 600.0, 780.0, 680.0), 0.86),
    ]
    source_predictions = non_maximum_suppression(simulated_tile_predictions, iou_threshold=0.5)

    masks = [
        np.pad(np.ones((512, 1024), dtype=bool), ((512, 0), (0, 0))),
        np.pad(np.ones((520, 1024), dtype=bool), ((504, 0), (0, 0))),
        np.pad(np.ones((500, 1024), dtype=bool), ((524, 0), (0, 0))),
    ]
    substrate = majority_consensus(masks)
    scale_record = {
        "image_id": "synthetic_001",
        "scale_bar_length_px": 250,
        "scale_bar_value": 50,
        "scale_bar_unit": "nm",
        "verification_status": "verified",
    }
    nm_per_px = calibrated_nm_per_pixel(scale_record)
    radius_px = 50.0
    contour = [
        (256.0 + radius_px * math.cos(angle), 256.0 + radius_px * math.sin(angle))
        for angle in np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
    ]
    metrics = projected_particle_metrics(math.pi * radius_px**2, contour, nm_per_px)
    return {
        "example_scope": "generated software-interface test; not a scientific accuracy benchmark",
        "source_image": {"image_id": "synthetic_001", "width_px": width, "height_px": height},
        "window_starts_px": starts,
        "merged_detection_count": len(source_predictions),
        "sam_prompt_interface": {
            "type": "xyxy_detector_boxes_in_source_coordinates",
            "prompt_count": len(source_predictions),
            "provider_checkpoint_included": False,
        },
        "substrate_consensus": {
            "member_count": len(masks),
            "required_votes": 2,
            "area_fraction": float(substrate.mean()),
        },
        "verified_scale_nm_per_px": nm_per_px,
        "projected_particle_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run_example(), indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
