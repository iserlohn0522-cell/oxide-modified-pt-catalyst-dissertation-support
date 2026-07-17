from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_model_registry_has_all_consensus_members_and_no_git_weights() -> None:
    registry = yaml.safe_load((ROOT / "models/MODEL_REGISTRY.yaml").read_text(encoding="utf-8"))
    models = registry["models"]
    assert len(models) == 4
    assert sum(item["task"] == "binary_semantic_segmentation" for item in models) == 3
    assert sum(item["task"] == "instance_segmentation" for item in models) == 1
    assert {item.get("threshold") for item in models if item["task"] == "binary_semantic_segmentation"} == {0.3, 0.5, 0.7}
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0 for item in models)
    assert not list(ROOT.rglob("*.pt"))


def test_third_party_provider_checkpoints_are_link_only() -> None:
    registry = yaml.safe_load((ROOT / "models/MODEL_REGISTRY.yaml").read_text(encoding="utf-8"))
    dependencies = {item["provider"]: item for item in registry["third_party_dependencies"]}
    assert dependencies["Ultralytics"]["redistributed"] is False
    assert dependencies["Meta"]["redistributed"] is False
    assert dependencies["Meta"]["license_string"] == "SAM License (custom)"
