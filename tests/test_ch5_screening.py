from __future__ import annotations

from pathlib import Path

from pt_tem_cv.screening import expand_augmentation_screen, expand_doe_screen, expand_family_screen
from pt_tem_cv.training import build_train_overrides


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "ch5_ml_tem"


def test_public_screen_configs_expand_deterministically() -> None:
    assert len(expand_family_screen(CONFIG_ROOT / "family_screen.yaml")) == 4
    assert len(expand_augmentation_screen(CONFIG_ROOT / "augmentation_screen.yaml")) == 14
    first = expand_doe_screen(CONFIG_ROOT / "doe_screen.yaml")
    second = expand_doe_screen(CONFIG_ROOT / "doe_screen.yaml")
    assert len(first) == 20
    assert first == second
    assert sum(run.augment_name == "baseline" for run in first) == 2
    assert sum(run.augment_name == "center" for run in first) == 2


def test_training_overrides_disable_color_and_composite_augmentation() -> None:
    overrides = build_train_overrides({"flip": True})
    assert overrides["fliplr"] == 0.5
    assert overrides["hsv_h"] == 0.0
    assert overrides["mosaic"] == 0.0
    assert overrides["mixup"] == 0.0
