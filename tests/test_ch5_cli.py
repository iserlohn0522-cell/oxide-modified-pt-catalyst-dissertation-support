from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_all_ch5_command_line_scripts_show_help() -> None:
    script_root = Path(__file__).resolve().parents[1] / "scripts" / "ch5_ml_tem"
    scripts = [path for path in script_root.glob("*.py") if not path.name.startswith("_")]
    assert scripts
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"
        assert "usage:" in result.stdout.lower()
