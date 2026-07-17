#!/usr/bin/env python3
"""Create a declared, path-free evidence excerpt from one CP2K input/output pair."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


INPUT_KEYS = (
    "RUN_TYPE",
    "CHARGE",
    "LSD",
    "MULTIPLICITY",
    "CUTOFF",
    "REL_CUTOFF",
    "NGRIDS",
    "TYPE DFTD3",
    "CALCULATE_C9_TERM",
    "PERIODIC",
    "PSOLVER",
)
OUTPUT_PATTERNS = (
    re.compile(r"CP2K\| version string:"),
    re.compile(r"GLOBAL\| Run type"),
    re.compile(r"CELL\| Periodicity"),
    re.compile(r"PW_GRID\| Cutoff"),
    re.compile(r"POISSON\| Solver"),
    re.compile(r"POISSON\| Periodicity"),
    re.compile(r"SCF run converged"),
    re.compile(r"ENERGY\| Total FORCE_EVAL"),
    re.compile(r"PROGRAM ENDED AT"),
)


def build_excerpt(input_path: Path, output_path: Path) -> str:
    input_lines = []
    for line in Path(input_path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("!") and any(stripped.startswith(key) for key in INPUT_KEYS):
            input_lines.append(stripped)

    output_lines = []
    cutoff_seen = False
    for line in Path(output_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not any(pattern.search(line) for pattern in OUTPUT_PATTERNS):
            continue
        stripped = line.strip()
        if stripped.startswith("PW_GRID| Cutoff"):
            if cutoff_seen:
                continue
            cutoff_seen = True
        output_lines.append(stripped)

    if not any("SCF run converged" in line for line in output_lines):
        raise ValueError("The selected output does not contain an SCF convergence marker")
    if not any("ENERGY| Total FORCE_EVAL" in line for line in output_lines):
        raise ValueError("The selected output does not contain a final total energy")
    if not any("PROGRAM ENDED AT" in line for line in output_lines):
        raise ValueError("The selected output does not contain a normal-termination marker")

    return "\n".join(
        [
            "# Declared CP2K evidence excerpt",
            "# Generated from the unpublished accepted input/output pair by extract_cp2k_output_excerpt.py.",
            "# This is not a complete CP2K output or scheduler log.",
            "",
            "[selected input settings]",
            *input_lines,
            "",
            "[selected output evidence]",
            *output_lines,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(build_excerpt(args.input, args.output), encoding="utf-8")
    print(args.destination)


if __name__ == "__main__":
    main()
