# Release Artifacts

The Git tree contains code, configurations, model cards, aggregate results, and curated DFT cases. Large custom weights remain separate GitHub Release assets so that the repository history stays reviewable.

Before loading an asset, compare its SHA-256 digest with both `models/MODEL_REGISTRY.yaml` and the release `SHA256SUMS.txt`. The four authoritative `v1.0.0` filenames are listed in the root README.

No archive containing the entire repository is treated as an additional authoritative scientific asset. SAM 3 and base YOLO provider weights are link-only third-party dependencies.
