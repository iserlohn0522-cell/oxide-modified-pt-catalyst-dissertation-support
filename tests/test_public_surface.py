from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_safety_scan():
    module_path = ROOT / "scripts" / "safety_scan.py"
    spec = importlib.util.spec_from_file_location("safety_scan", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_passes_public_release_scan() -> None:
    scanner = _load_safety_scan()
    assert scanner.scan_path(ROOT) == []


def test_scan_blocks_private_paths_weights_and_coordinates(tmp_path: Path) -> None:
    scanner = _load_safety_scan()
    (tmp_path / "model.pt").write_bytes(b"not a real model")
    (tmp_path / "structure.xyz").write_text("1\nexample\nPt 0 0 0\n", encoding="utf-8")
    private_path = "/".join(("D:", "projects", "private_research_workspace"))
    (tmp_path / "notes.md").write_text(private_path, encoding="utf-8")
    fake_token = "ghp_" + "A" * 24
    (tmp_path / "secrets.txt").write_text(fake_token, encoding="utf-8")

    failures = scanner.scan_path(tmp_path)

    assert any("model.pt" in failure for failure in failures)
    assert any("structure.xyz" in failure for failure in failures)
    assert any("blocked text regex" in failure for failure in failures)
