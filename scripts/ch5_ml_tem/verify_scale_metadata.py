from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from pt_tem_cv.scale import calibrated_nm_per_pixel


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate human-reviewed scale records and add nm_per_px.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_rows = [{**row, "nm_per_px": f"{calibrated_nm_per_pixel(row):.12g}"} for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0]) if output_rows else ["image_id", "scale_bar_length_px", "scale_bar_value", "scale_bar_unit", "verification_status", "nm_per_px"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)


if __name__ == "__main__":
    main()
