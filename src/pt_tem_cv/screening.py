from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ScreenRun:
    run_id: str
    family: str
    model: str
    augment_name: str
    train_transforms: dict[str, Any]
    factor_codes: dict[str, int]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def expand_family_screen(config_path: Path) -> list[ScreenRun]:
    raw = _load_yaml(Path(config_path))
    version = str(raw.get("version", "family_screen"))
    return [
        ScreenRun(f"{version}/{family}", str(family), str(model), "baseline", {}, {})
        for family, model in dict(raw.get("families") or {}).items()
    ]


def expand_augmentation_screen(config_path: Path) -> list[ScreenRun]:
    raw = _load_yaml(Path(config_path))
    version = str(raw.get("version", "augmentation_screen"))
    runs: list[ScreenRun] = []
    for family, model in dict(raw.get("families") or {}).items():
        for augment_name, transforms in dict(raw.get("augment_packs") or {}).items():
            runs.append(
                ScreenRun(
                    f"{version}/{family}/{augment_name}",
                    str(family),
                    str(model),
                    str(augment_name),
                    dict(transforms or {}),
                    {},
                )
            )
    return runs


def _sample_seed(base_seed: int, family: str, arm: str) -> int:
    digest = hashlib.blake2b(f"{base_seed}:{family}:{arm}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**31)


def expand_doe_screen(config_path: Path) -> list[ScreenRun]:
    raw = _load_yaml(Path(config_path))
    version = str(raw.get("version", "doe_screen"))
    base_seed = int(raw.get("random_seed", 0))
    factor_levels = dict(raw.get("factor_levels") or {})
    factor_names = list(factor_levels)
    if not factor_names:
        raise ValueError("DOE config must define factor_levels")
    runs: list[ScreenRun] = []
    for family, model in dict(raw.get("families") or {}).items():
        family_runs: list[ScreenRun] = [
            ScreenRun(f"{version}/{family}/baseline", str(family), str(model), "baseline", {}, {})
        ]
        for codes in product((-1, 1), repeat=len(factor_names)):
            code_map = {name: int(code) for name, code in zip(factor_names, codes)}
            levels = {
                name: float(dict(factor_levels[name])["low" if code < 0 else "high"])
                for name, code in code_map.items()
            }
            arm = "__".join(f"{name}_{'low' if code < 0 else 'high'}" for name, code in code_map.items())
            family_runs.append(
                ScreenRun(
                    f"{version}/{family}/{arm}",
                    str(family),
                    str(model),
                    arm,
                    {**levels, "seed": _sample_seed(base_seed, str(family), arm)},
                    code_map,
                )
            )
        center = {name: float(dict(factor_levels[name])["center"]) for name in factor_names}
        family_runs.append(
            ScreenRun(
                f"{version}/{family}/center",
                str(family),
                str(model),
                "center",
                {**center, "seed": _sample_seed(base_seed, str(family), "center")},
                {name: 0 for name in factor_names},
            )
        )
        baseline, randomized = family_runs[0], family_runs[1:]
        random.Random(f"{base_seed}:{family}").shuffle(randomized)
        runs.extend([baseline, *randomized])
    return runs


__all__ = ["ScreenRun", "expand_augmentation_screen", "expand_doe_screen", "expand_family_screen"]
