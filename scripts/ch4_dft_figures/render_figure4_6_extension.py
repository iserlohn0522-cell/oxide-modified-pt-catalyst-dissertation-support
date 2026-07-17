#!/usr/bin/env python3
"""Render dissertation Figure 4.6 from the public plot-level CSV."""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPOSITORY_ROOT / "data" / "ch4_dft" / "figure4_6_extension.csv"
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "oxide_modified_pt_dissertation_support" / "ch4_dft"
EXPECTED_ENDPOINTS = ("TaOx*", "Zr-JC*", "Zr-JC", "TiOx", "Nb-WP", "CeOx", "WOx")
WIDTH_PX = 1676
HEIGHT_PX = 1068
DPI = 220


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate the endpoint records required by Figure 4.6."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "oxide_family",
            "endpoint_label",
            "delta_e_sep_2A_eV",
            "method_family",
            "evidence_role",
            "note",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing required columns in {path}")
        rows = {row["endpoint_label"]: row for row in reader}

    if set(rows) != set(EXPECTED_ENDPOINTS):
        raise ValueError(f"Expected endpoint labels {EXPECTED_ENDPOINTS}; found {tuple(rows)}")
    if rows["CeOx"]["delta_e_sep_2A_eV"].strip():
        raise ValueError("CeOx must remain unquantified in the public extension figure")
    if "different relaxation protocol" not in rows["TaOx*"]["method_family"].lower():
        raise ValueError("TaOx* must be identified as a different-protocol value")
    return rows


def numeric_value(rows: dict[str, dict[str, str]], endpoint: str) -> float:
    return float(rows[endpoint]["delta_e_sep_2A_eV"])


def render(data_path: Path, output_dir: Path) -> list[Path]:
    rows = load_rows(data_path)
    labels = [r"TaO$_x$*", r"ZrO$_x$", r"TiO$_x$", r"NbO$_x$", r"CeO$_x$", r"WO$_x$"]
    values = {
        r"TaO$_x$*": numeric_value(rows, "TaOx*"),
        r"TiO$_x$": numeric_value(rows, "TiOx"),
        r"NbO$_x$": numeric_value(rows, "Nb-WP"),
        r"WO$_x$": numeric_value(rows, "WOx"),
    }
    zr_low = numeric_value(rows, "Zr-JC*")
    zr_high = numeric_value(rows, "Zr-JC")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    primary = "#2F7F83"
    extension = "#72787D"
    fig, ax = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for y, label in enumerate(labels):
        if label == r"CeO$_x$":
            ax.scatter(0, y, marker="x", s=75, linewidth=1.8, color=extension, zorder=4)
            ax.text(0.22, y, "not evaluated", va="center", ha="left", color=extension, fontsize=11)
            continue
        if label == r"ZrO$_x$":
            ax.hlines(y, 0, zr_high, color="#C8CDD1", linewidth=3.0, zorder=1)
            ax.hlines(y, zr_low, zr_high, color="#20282A", linewidth=2.0, zorder=5)
            ax.scatter([zr_low, zr_high], [y, y], s=45, color="#20282A", zorder=6)
            ax.scatter(zr_high, y, s=80, facecolor="#20282A", edgecolor=primary, linewidth=1.6, zorder=7)
            ax.text(zr_low - 0.12, y - 0.23, f"{zr_low:.2f}", ha="right", va="center", fontsize=10.5)
            ax.text(zr_high + 0.18, y, f"{zr_high:.2f}", va="center", ha="left", fontsize=11.5)
            continue

        value = values[label]
        color = primary if label == r"NbO$_x$" else extension
        ax.hlines(y, min(0, value), max(0, value), color="#C8CDD1", linewidth=3.0, zorder=1)
        marker = "D" if label == r"TaO$_x$*" else "o"
        facecolor = "white" if label == r"TaO$_x$*" else color
        ax.scatter(value, y, marker=marker, s=80, facecolor=facecolor, edgecolor=color, linewidth=1.6, zorder=4)
        ax.text(
            value + (0.18 if value >= 0 else -0.18),
            y,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=11.5,
        )

    ax.axvline(0, color="#20282A", linewidth=1.1)
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-2.55, 8.45)
    ax.set_xlabel(r"$\Delta E_{\mathrm{sep}}(2\ \mathrm{\AA})$ (eV)")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#20282A")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=4, width=0.9)
    fig.subplots_adjust(left=0.14, right=0.965, top=0.965, bottom=0.16)

    outputs: list[Path] = []
    for extension_name in ("png", "svg", "pdf"):
        output = output_dir / f"Figure_4_6_extension.{extension_name}"
        fig.savefig(output, dpi=DPI if extension_name == "png" else None, facecolor="white")
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
