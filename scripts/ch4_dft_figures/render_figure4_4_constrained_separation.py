#!/usr/bin/env python3
"""Render dissertation Figure 4.4 from the public plot-level CSV."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPOSITORY_ROOT / "data" / "ch4_dft" / "figure4_4_constrained_separation.csv"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "oxide_modified_pt_dissertation_support" / "ch4_dft"
EXPECTED_ORDER = ("Nb-WP", "Zr-JC*", "Zr-JC")
DISPLAY_NOTES = {
    "Nb-WP": "Weak-contact\ncomparator",
    "Zr-JC*": "Second lower-energy\ndisplaced basin",
    "Zr-JC": "Primary displaced\nbasin",
}
COLORS = {"Nb-WP": "#b66f4f", "Zr-JC*": "#8abdc2", "Zr-JC": "#1f6f78"}


def load_values(path: Path) -> dict[str, float]:
    """Load and validate the three endpoint values required by Figure 4.4."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"endpoint_label", "delta_e_sep_2A_eV", "method_family", "evidence_role", "note"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}")
        rows = list(reader)

    values = {row["endpoint_label"]: float(row["delta_e_sep_2A_eV"]) for row in rows}
    if set(values) != set(EXPECTED_ORDER):
        raise ValueError(f"Expected endpoint labels {EXPECTED_ORDER}; found {tuple(values)}")
    return values


def add_bracket(ax, x0: float, x1: float, y: float, label: str, color: str) -> None:
    ax.plot([x0, x0, x1, x1], [y - 0.07, y, y, y - 0.07], color=color, linewidth=1.35)
    ax.text((x0 + x1) / 2, y + 0.07, label, ha="center", va="bottom", color=color, fontsize=9.2)


def render(data_path: Path, output_dir: Path) -> list[Path]:
    values_by_label = load_values(data_path)
    labels = list(EXPECTED_ORDER)
    values = [values_by_label[label] for label in labels]
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300, constrained_layout=True)
    x_positions = list(range(len(labels)))
    bars = ax.bar(
        x_positions,
        values,
        width=0.58,
        color=[COLORS[label] for label in labels],
        edgecolor="#303030",
        linewidth=0.7,
    )
    for bar, label, value in zip(bars, labels, values):
        xpos = bar.get_x() + bar.get_width() / 2
        ax.text(xpos, value + 0.075, f"{value:.2f}", ha="center", va="bottom", color="#252525", fontsize=10)
        ax.text(
            xpos,
            -0.38,
            DISPLAY_NOTES[label],
            ha="center",
            va="top",
            color="#2e2e2e",
            fontsize=8.8,
            linespacing=1.05,
        )

    ax.set_ylabel(r"$\Delta E_{\mathrm{sep}}$ (2 $\mathrm{\AA}$), eV")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.72, 3.55)
    ax.grid(False)

    lower_margin = values_by_label["Zr-JC*"] - values_by_label["Nb-WP"]
    primary_margin = values_by_label["Zr-JC"] - values_by_label["Nb-WP"]
    add_bracket(ax, 0, 1, 1.73, f"Zr-JC* - Nb-WP = +{lower_margin:.2f} eV", "#4f8f96")
    add_bracket(ax, 0, 2, 3.20, f"Zr-JC - Nb-WP = +{primary_margin:.2f} eV", "#1f6f78")

    outputs: list[Path] = []
    for extension in ("png", "svg", "pdf"):
        output = output_dir / f"Figure_4_4_constrained_separation.{extension}"
        fig.savefig(output, dpi=300 if extension == "png" else None, facecolor="white")
        outputs.append(output)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Input CSV (default: repository data file).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG, SVG, and PDF outputs (default: system temporary directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output in render(args.data, args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
