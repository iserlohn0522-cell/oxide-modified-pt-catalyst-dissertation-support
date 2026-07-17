from __future__ import annotations

import argparse
import re
from pathlib import Path


BLOCKED_EXTENSIONS = {
    ".7z",
    ".ckpt",
    ".czi",
    ".dm3",
    ".dm4",
    ".doc",
    ".docx",
    ".emd",
    ".emi",
    ".engine",
    ".gz",
    ".jpeg",
    ".jpg",
    ".mrc",
    ".nd2",
    ".onnx",
    ".pdf",
    ".pem",
    ".png",
    ".pt",
    ".pth",
    ".rar",
    ".ser",
    ".tar",
    ".tif",
    ".tiff",
    ".xyz",
    ".zip",
}

BLOCKED_REGEX = [
    re.compile(r"\b[A-Za-z]:[\\/](?:Users|projects)[\\/]", re.IGNORECASE),
    re.compile(r"/(?:Users|home)/[^/\s]+/", re.IGNORECASE),
    re.compile(r"\bjob\s+\d{6,}\b", re.IGNORECASE),
    re.compile(r"\b(?:T[B]D|TO[D]O|FIX[M]E|PLACEHOLD[E]R)\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b" + "ghp_" + r"[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b" + "github_pat_" + r"[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b" + "sk" + r"-[A-Za-z0-9_-]{20,}\b"),
]

SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TEXT_EXTENSIONS = {".cff", ".csv", ".gitignore", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def _skip(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts)


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".gitignore", "LICENSE"}


def scan_path(root: Path, max_file_mb: float = 2.0) -> list[str]:
    root = Path(root).resolve()
    max_bytes = int(max_file_mb * 1024 * 1024)
    failures: list[str] = []

    for path in root.rglob("*"):
        if _skip(path, root) or path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()

        if suffix in BLOCKED_EXTENSIONS:
            failures.append(f"blocked extension: {rel}")
        if path.stat().st_size > max_bytes:
            failures.append(f"file too large: {rel}")

        if _is_text_file(path):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in BLOCKED_REGEX:
                if pattern.search(text):
                    failures.append(f"blocked text regex {pattern.pattern!r}: {rel}")

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the public repository for excluded files and internal paths.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--max-file-mb", type=float, default=2.0)
    args = parser.parse_args()
    failures = scan_path(args.root, max_file_mb=args.max_file_mb)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("Public-release safety scan passed")


if __name__ == "__main__":
    main()
