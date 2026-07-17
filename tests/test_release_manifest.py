import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_manifest_file_hashes() -> None:
    manifest = json.loads((ROOT / "manifests/RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["release"] == "v1.0.0"
    assert manifest["file_count"] == len(manifest["files"])
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
