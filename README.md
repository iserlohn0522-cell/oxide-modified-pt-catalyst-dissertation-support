# Oxide-Modified Pt Catalyst Dissertation Support

This repository is the stable support package for Gan Xu's 2026 doctoral dissertation, *Interfacial Engineering of Metal Oxide-Modified Pt-Based Catalysts for Durable PEMFC Cathodes* (University of Missouri-Columbia). It releases the reusable Chapter 5 TEM computer-vision workflow, sanitized custom model weights as separate release assets, and a curated Chapter 4 DFT reproducibility subset. It is not a complete research-data archive.

## Contents

- `src/pt_tem_cv/`: sliding-window dataset preparation, controlled augmentation, detector training/evaluation, substrate-consensus helpers, verified scale conversion, and projected two-dimensional morphometry.
- `scripts/ch5_ml_tem/` and `configs/ch5_ml_tem/`: runnable entry points and production-facing public configurations.
- `models/`: model registry, SHA-256 digests, model cards, and third-party model boundaries. Custom weights are release assets, not Git objects.
- `data/ch5_ml_tem/aggregate_results.csv`: aggregate dissertation results without image identities or per-particle records.
- `data/ch4_dft/`: five curated accepted calculation cases, endpoint energies, protocol classes, and figure tables.
- `scripts/ch4_dft/` and `scripts/ch4_dft_figures/`: declared CP2K excerpt/energy extraction and figure reconstruction.
- `examples/synthetic_tem/`: generated software-interface test, not an accuracy benchmark.
- `tests/`: data-free regression, reconstruction, privacy, and public-surface checks.

## Installation

Python 3.10 or later is required.

```text
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/safety_scan.py .
```

Detector training/loading additionally uses the `ml` extra. Substrate-model inference uses the `substrate` extra.

```text
python -m pip install -e ".[ml,substrate]"
```

## Model release assets

Version `v1.0.0` uses four deliberately separate assets:

- `particle_detector_yolov8m_seg_dissertation_v1.pt`
- `substrate_unet_resnet34_seed20260623_v1.pt`
- `substrate_unet_resnet34_seed20260624_v1.pt`
- `substrate_unet_resnet34_seed20260625_v1.pt`

Download them deliberately from the matching versioned GitHub Release and verify them against `models/MODEL_REGISTRY.yaml` or the release `SHA256SUMS.txt`. The detector asset is an Ultralytics pickle-based checkpoint; load only the named asset whose digest matches the registry. The three substrate assets are pure state dictionaries loadable with `torch.load(..., weights_only=True)`.

The weights were sanitized outside Git. Model parameters were tensor-identical to the accepted source models, and fixed-input forward comparisons gave a maximum absolute difference of 0.0. The release does not include the SAM 3 provider checkpoint or a base YOLO checkpoint; obtain third-party material only from its provider under its current terms.

## Chapter 5 scope

The public workflow preserves the scientific separation between particle detection, detector-box-prompted mask refinement, substrate segmentation, scale verification, and projected morphometry. The final particle-mask route passed detector boxes to SAM 3; the provider checkpoint and private production masks are not distributed. The three public substrate checkpoints are combined by a two-of-three vote using the recorded per-seed thresholds.

Scale-bar geometry and associated text were read computationally and then checked by a human before conversion to physical units. Tiles and rotations are computational units; source TEM images and material states remain the measurement units. Reported area, perimeter, equivalent diameter, and circularity are projected two-dimensional descriptors, not reconstructions of three-dimensional particle volume or surface area.

The public aggregate table records the final evaluation and deployment counts without releasing image identities, split membership, masks, or per-particle records. Detailed interfaces and limitations are in `docs/ch5_ml_tem/README.md`.

## Chapter 4 scope

The DFT package releases the selected Zr-JC, Zr-JC*, Nb-WP, Zr-OS, and Nb-JC endpoint cases with sanitized CP2K inputs, coordinates, declared output excerpts, accepted energies, and hashes linking each excerpt to its unpublished accepted full output. It also records the Ti/W extension energies used in the figure table.

Protocol classes remain explicit. The principal Zr/Nb comparison uses 600/50 Ry, NGRIDS 5, and nonperiodic electrostatics. Zr-OS and Nb-JC form a separate matched 400/50 Ry, NGRIDS 4, periodic control comparison. WOx retains a periodic cell at 600/50 Ry. No public script compares raw absolute energies across these incompatible protocol classes. See `docs/ch4_dft/README.md`.

## Reproducibility and release boundary

The synthetic example checks interfaces only. Scientific accuracy is represented by aggregate dissertation results, not by generated data. Original TEM images, editable annotations, frozen split identities, SAM 3/base-YOLO weights, production predictions and masks, per-particle records, electrochemical raw data, full DFT outputs, wavefunctions, scheduler files, internal logs, and dissertation submission files are excluded.

See `DATA_AVAILABILITY.md`, `docs/REPRODUCIBILITY.md`, and `docs/PRIVACY_AND_EXCLUSIONS.md` for the exact boundary.

## Citation

For code, curated data, or release assets, cite the versioned repository release:

> Xu, G. *Oxide-Modified Pt Catalyst Dissertation Support*, version 1.0.0, 2026. https://github.com/iserlohn0522-cell/oxide-modified-pt-catalyst-dissertation-support

For scientific results and interpretation, cite the dissertation:

> Xu, G. *Interfacial Engineering of Metal Oxide-Modified Pt-Based Catalysts for Durable PEMFC Cathodes*. Ph.D. dissertation, University of Missouri-Columbia, 2026.

Machine-readable metadata are provided in `CITATION.cff`.

## Licenses

The author's original software, documentation, and custom model assets are licensed under AGPL-3.0-or-later. The author's curated numerical tables and coordinate data are licensed under CC BY 4.0. Third-party software and models remain under their providers' terms. See `NOTICE.md`, `DATA_LICENSE.md`, and `THIRD_PARTY_NOTICES.md`.
