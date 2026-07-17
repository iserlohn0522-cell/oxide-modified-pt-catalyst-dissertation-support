#!/usr/bin/env python3
"""Extract the final total FORCE_EVAL energy from a CP2K output file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ENERGY_PATTERN = re.compile(r"ENERGY\|\s+Total FORCE_EVAL .* energy \[a\.u\.\]:\s+([-+0-9.Ee]+)")


def extract_final_energy(path: Path) -> float:
    matches = ENERGY_PATTERN.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"No CP2K FORCE_EVAL energy found in {path}")
    return float(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"{extract_final_energy(args.output):.15f}")


if __name__ == "__main__":
    main()
