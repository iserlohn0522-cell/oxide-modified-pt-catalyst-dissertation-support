# Reproducibility Guide

## Software checks

Create a clean Python 3.10–3.12 environment, install `.[dev]`, and run `python -m pytest -q`. The tests reconstruct DFT energy differences, validate protocol boundaries and hashes, render the released figure tables, exercise the synthetic Chapter 5 interfaces, and run the public-release safety scan.

The synthetic example is intentionally independent of the dissertation image set. It confirms data flow and schema behavior, not scientific accuracy.

## Chapter 5 models

Download the four assets from the release matching the repository version. Check the filename, byte count, and SHA-256 digest against `models/MODEL_REGISTRY.yaml`. Use Ultralytics 8.4.9 for the detector asset and `segmentation-models-pytorch` 0.5.0 for the substrate state dictionaries. The exact inference preprocessing and member thresholds are in `configs/ch5_ml_tem/`.

The released weights reproduce the accepted model parameters. They do not supply the private microscopy data needed to reproduce the dissertation's accuracy estimates from scratch.

## Chapter 4 DFT

Each case under `data/ch4_dft/cases/` contains sanitized CP2K input/coordinate files, declared evidence excerpts, endpoint results, and hashes. CP2K 2023.1 was used for the released accepted outputs. The full output hashes in `result.json` identify the unpublished source files from which the excerpts were generated.

Reconstruction tests calculate `E(displaced) - E(retained)` with the explicit Hartree-to-eV conversion. Only endpoints within the same `protocol_id` are combined. The figure scripts use the final dissertation-selected difference tables and preserve method-distinct/unevaluated entries.
