import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARTREE_TO_EV = 27.211386245988


def test_endpoint_differences_match_public_table() -> None:
    endpoint_rows = list(csv.DictReader((ROOT / "data/ch4_dft/selected_endpoint_energies.csv").open(encoding="utf-8")))
    energies = {(row["pair_id"], row["endpoint_role"]): float(row["energy_Ha"]) for row in endpoint_rows}
    pair_rows = list(csv.DictReader((ROOT / "data/ch4_dft/selected_pair_differences.csv").open(encoding="utf-8")))
    for row in pair_rows:
        pair_id = row["pair_id"]
        if (pair_id, "retained") not in energies:
            continue
        calculated = (energies[(pair_id, "displaced")] - energies[(pair_id, "retained")]) * HARTREE_TO_EV
        assert calculated == pytest.approx(float(row["delta_e_displaced_minus_retained_eV"]), abs=5e-6)


def test_curated_case_files_match_result_hashes() -> None:
    for result_path in (ROOT / "data/ch4_dft/cases").glob("*/*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        endpoint = result_path.parent
        for filename, expected_hash in result["public_file_sha256"].items():
            assert hashlib.sha256((endpoint / filename).read_bytes()).hexdigest() == expected_hash


def test_method_boundaries_are_explicit() -> None:
    rows = {row["pair_id"]: row for row in csv.DictReader((ROOT / "data/ch4_dft/selected_pair_differences.csv").open(encoding="utf-8"))}
    assert rows["Zr-JC"]["protocol_id"] == "M1_600Ry_nonperiodic"
    assert rows["Zr-OS"]["protocol_id"] == rows["Nb-JC"]["protocol_id"] == "M2_400Ry_periodic"
    assert rows["WOx"]["protocol_id"] == "M3_600Ry_periodic"
    assert rows["TaOx*"]["protocol_id"] == "M4_method_distinct"


def test_figure_tables_rebuild_from_selected_pairs() -> None:
    pairs = {
        row["pair_id"]: row
        for row in csv.DictReader((ROOT / "data/ch4_dft/selected_pair_differences.csv").open(encoding="utf-8"))
    }
    figure_44 = {
        row["endpoint_label"]: row
        for row in csv.DictReader((ROOT / "data/ch4_dft/figure4_4_constrained_separation.csv").open(encoding="utf-8"))
    }
    for pair_id in ("Zr-JC", "Zr-JC*", "Nb-WP"):
        assert float(figure_44[pair_id]["delta_e_sep_2A_eV"]) == float(pairs[pair_id]["delta_e_displaced_minus_retained_eV"])

    figure_46 = {
        row["endpoint_label"]: row
        for row in csv.DictReader((ROOT / "data/ch4_dft/figure4_6_extension.csv").open(encoding="utf-8"))
    }
    for endpoint, pair_id in (("TaOx*", "TaOx*"), ("Zr-JC*", "Zr-JC*"), ("Zr-JC", "Zr-JC"), ("TiOx", "TiOx"), ("Nb-WP", "Nb-WP"), ("WOx", "WOx")):
        assert float(figure_46[endpoint]["delta_e_sep_2A_eV"]) == float(pairs[pair_id]["delta_e_displaced_minus_retained_eV"])
    assert figure_46["CeOx"]["delta_e_sep_2A_eV"] == ""
    assert "different relaxation protocol" in figure_46["TaOx*"]["method_family"]
    assert "periodic" in figure_46["WOx"]["method_family"]
