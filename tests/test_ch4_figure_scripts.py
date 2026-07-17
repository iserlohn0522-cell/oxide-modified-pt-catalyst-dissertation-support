from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPOSITORY_ROOT / "scripts" / "ch4_dft_figures"
DATA_DIR = REPOSITORY_ROOT / "data" / "ch4_dft"


def run_figure_script(script_name: str, output_dir: Path) -> None:
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), "--output-dir", str(output_dir)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_default_output_uses_system_temporary_directory(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.update({"MPLBACKEND": "Agg", "TEMP": str(tmp_path), "TMP": str(tmp_path), "TMPDIR": str(tmp_path)})
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "render_figure4_4_constrained_separation.py")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    default_output = tmp_path / "oxide_modified_pt_dissertation_support" / "ch4_dft"
    assert (default_output / "Figure_4_4_constrained_separation.png").is_file()
    assert not list(SCRIPT_DIR.glob("Figure_4_4_constrained_separation.*"))


def test_figure_4_4_renders_in_requested_temporary_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "figure_4_4"
    run_figure_script("render_figure4_4_constrained_separation.py", output_dir)

    for extension in ("png", "svg", "pdf"):
        output = output_dir / f"Figure_4_4_constrained_separation.{extension}"
        assert output.is_file()
        assert output.stat().st_size > 0


def test_figure_4_6_renders_in_requested_temporary_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "figure_4_6"
    run_figure_script("render_figure4_6_extension.py", output_dir)

    for extension in ("png", "svg", "pdf"):
        output = output_dir / f"Figure_4_6_extension.{extension}"
        assert output.is_file()
        assert output.stat().st_size > 0


def test_public_ch4_figure_csvs_contain_only_plot_level_fields() -> None:
    allowed_fields = {
        "oxide_family",
        "endpoint_label",
        "delta_e_sep_2A_eV",
        "method_family",
        "evidence_role",
        "note",
    }
    forbidden_fragments = (
        "energy_ha",
        "retained_energy",
        "displaced_energy",
    )

    for csv_path in (
        DATA_DIR / "figure4_4_constrained_separation.csv",
        DATA_DIR / "figure4_6_extension.csv",
    ):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None
            assert set(reader.fieldnames) <= allowed_fields
            text = csv_path.read_text(encoding="utf-8").lower()
            assert all(fragment not in text for fragment in forbidden_fragments)
            assert not re.search(r"\b[A-Za-z]:[\\/]", text)
            assert not re.search(r"\bjob\s+\d{6,}\b", text)


def test_scripts_are_utf8_without_replacement_characters_or_absolute_paths() -> None:
    for script in SCRIPT_DIR.glob("*.py"):
        text = script.read_text(encoding="utf-8")
        assert "\ufffd" not in text
        assert not re.search(r"\b[A-Za-z]:[\\/]", text)
