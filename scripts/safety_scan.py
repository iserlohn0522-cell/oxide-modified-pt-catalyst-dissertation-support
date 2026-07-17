from __future__ import annotations

import argparse
import re
import subprocess
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
    ".inp",
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
    ".restart",
    ".sbatch",
    ".ser",
    ".tar",
    ".tif",
    ".tiff",
    ".wfn",
    ".xyz",
    ".zip",
}

ALLOWED_EXTENSION_PATHS = {
    ".inp": ("data/ch4_dft/cases/",),
    ".xyz": ("data/ch4_dft/cases/",),
}

BLOCKED_REGEX = [
    re.compile(r"\b[A-Za-z]:[\\/](?:Users|projects)[\\/]", re.IGNORECASE),
    re.compile(r"/(?:Users|home)/[^/\s]+/", re.IGNORECASE),
    re.compile(r"/(?:cluster|scratch)/[^\s]*", re.IGNORECASE),
    re.compile(r"/(?:mnt/)?pixstor/[^\s]*", re.IGNORECASE),
    re.compile(r"\b(?:OneDrive|\.codex)[\\/]", re.IGNORECASE),
    re.compile(r"\b(?:gx" + "zmd|Gan_v[36]|V5" + "g|S2" + "LSD)\b", re.IGNORECASE),
    re.compile(r"\bjob\s+\d{6,}\b", re.IGNORECASE),
    re.compile(r"\b\d{7,9}\b(?=\s*(?:job|slurm))", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b(?:T[B]D|TO[D]O|FIX[M]E|PLACEHOLD[E]R)\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b" + "ghp_" + r"[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b" + "github_pat_" + r"[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b" + "sk" + r"-[A-Za-z0-9_-]{20,}\b"),
]

SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}
TEXT_EXTENSIONS = {".cff", ".csv", ".gitignore", ".inp", ".json", ".md", ".py", ".toml", ".txt", ".xyz", ".yaml", ".yml"}


def _skip(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    return any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in parts)


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".gitignore", "LICENSE"}


def _extension_is_allowed(rel: str, suffix: str) -> bool:
    return any(rel.startswith(prefix) for prefix in ALLOWED_EXTENSION_PATHS.get(suffix, ()))


def _scan_text(text: str, rel: str) -> list[str]:
    return [f"blocked text regex {pattern.pattern!r}: {rel}" for pattern in BLOCKED_REGEX if pattern.search(text)]


def scan_path(root: Path, max_file_mb: float = 2.0) -> list[str]:
    root = Path(root).resolve()
    max_bytes = int(max_file_mb * 1024 * 1024)
    failures: list[str] = []

    for path in root.rglob("*"):
        if _skip(path, root) or path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()

        if suffix in BLOCKED_EXTENSIONS and not _extension_is_allowed(rel, suffix):
            failures.append(f"blocked extension: {rel}")
        if path.stat().st_size > max_bytes:
            failures.append(f"file too large: {rel}")

        if _is_text_file(path):
            text = path.read_text(encoding="utf-8", errors="ignore")
            failures.extend(_scan_text(text, rel))

    return failures


def scan_git_history(root: Path, max_file_mb: float = 2.0) -> list[str]:
    root = Path(root).resolve()
    max_bytes = int(max_file_mb * 1024 * 1024)
    failures: list[str] = []
    commits = subprocess.run(
        ["git", "rev-list", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    for commit in commits:
        paths = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit], cwd=root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        for rel in paths:
            suffix = Path(rel).suffix.lower()
            label = f"{commit[:12]}:{rel}"
            if suffix in BLOCKED_EXTENSIONS and not _extension_is_allowed(rel, suffix):
                failures.append(f"blocked history extension: {label}")
            size = int(
                subprocess.run(
                    ["git", "cat-file", "-s", f"{commit}:{rel}"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
            if size > max_bytes:
                failures.append(f"history file too large: {label}")
            if suffix in TEXT_EXTENSIONS or Path(rel).name in {".gitignore", "LICENSE"}:
                text = subprocess.run(
                    ["git", "show", f"{commit}:{rel}"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    errors="replace",
                ).stdout
                failures.extend(_scan_text(text, label))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the public repository for excluded files and internal paths.")
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--max-file-mb", type=float, default=2.0)
    parser.add_argument("--git-history", action="store_true")
    args = parser.parse_args()
    failures = scan_path(args.root, max_file_mb=args.max_file_mb)
    if args.git_history:
        failures.extend(scan_git_history(args.root, max_file_mb=args.max_file_mb))
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("Public-release safety scan passed")


if __name__ == "__main__":
    main()
