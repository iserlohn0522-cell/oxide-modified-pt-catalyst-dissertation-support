import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_file_hashes_in_source_mapping() -> None:
    rows = list(csv.DictReader((ROOT / "manifests/SOURCE_TO_PUBLIC_MAPPING.csv").open(encoding="utf-8")))
    for row in rows:
        public_artifact = row["public_artifact"]
        if public_artifact.startswith("release asset "):
            continue
        path = ROOT / public_artifact
        assert path.is_file(), public_artifact
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["public_sha256"]
