import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_minimal_example_interfaces() -> None:
    path = ROOT / "examples/synthetic_tem/run_minimal_example.py"
    spec = importlib.util.spec_from_file_location("synthetic_tem_example", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.run_example()
    assert result["merged_detection_count"] == 2
    assert result["substrate_consensus"]["area_fraction"] == pytest.approx(0.5)
    assert result["verified_scale_nm_per_px"] == pytest.approx(0.2)
    assert result["sam_prompt_interface"]["provider_checkpoint_included"] is False
    assert result["projected_particle_metrics"]["equivalent_diameter_nm"] == pytest.approx(20.0)
