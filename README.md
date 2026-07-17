# Oxide-Modified Pt Catalyst Dissertation Support

This repository is the stable, lightweight support package for Gan Xu's 2026 doctoral dissertation, *Interfacial Engineering of Metal Oxide-Modified Pt-Based Catalysts for Durable PEMFC Cathodes* (University of Missouri-Columbia).

It contains the Chapter 5 detector training and evaluation code and a small set of Chapter 4 plotting and provenance files referenced by the dissertation. It is not a complete research-data archive.

## Repository contents

- `src/pt_tem_cv/`: reusable Chapter 5 code for sliding-window dataset construction, controlled augmentation, detector training, and source-image-level evaluation.
- `scripts/ch5_ml_tem/`: command-line entry points for the Chapter 5 code.
- `configs/ch5_ml_tem/`: public configuration templates. Dataset paths and split identities are intentionally user supplied.
- `scripts/ch4_dft_figures/`: plotting scripts corresponding to the selected Chapter 4 endpoint-energy figures.
- `data/ch4_dft/`: compact provenance tables containing values already reported in the dissertation.
- `docs/ch4_dft/` and `docs/ch5_ml_tem/`: concise file formats, method scope, and interpretation notes.
- `tests/`: data-free tests that generate any temporary fixtures at runtime.

## Installation and checks

Python 3.10 or later is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/safety_scan.py .
```

Detector training additionally requires the optional machine-learning dependencies:

```bash
python -m pip install -e ".[ml]"
```

The repository does not download research images, annotations, or model weights. Ultralytics model weights requested by a user may be downloaded under Ultralytics' own terms.

## Chapter 5 scope

The public code preserves the detector-side workflow used to create overlapping image windows, apply controlled train-time transformations, compare model or augmentation settings, train Ultralytics detectors, and return predictions to source-image coordinates for evaluation. Configuration and manifest examples show the required schemas without revealing the dissertation's image identities or frozen split membership.

The dissertation's production particle-mask step passed detector boxes to SAM 3 through the Ultralytics interface. The released code covers detector-side training and evaluation; production masks and weights are not distributed. Input formats and commands are documented in [docs/ch5_ml_tem/README.md](docs/ch5_ml_tem/README.md).

For scientific interpretation, the source TEM image and material state remain the measurement units; tiles and rotations are computational units. Reported morphometry is projected two-dimensional morphometry, not a reconstruction of three-dimensional particle size, volume, surface area, or support loading. Physical-unit conversion in the dissertation was performed only after image-by-image scale verification.

## Chapter 4 scope

The Chapter 4 package contains only compact values and plotting code for the selected constrained-separation comparison and oxide-fragment extension shown in the final dissertation. Full calculation archives and coordinate sets remain outside the public package. The TaOx* entry remains explicitly identified as method-distinct, and CeOx is not assigned a quantitative value.

The data columns and plotting commands are documented in [docs/ch4_dft/README.md](docs/ch4_dft/README.md).

## Data availability

The public-release boundary is described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). In particular, the repository excludes original microscopy images, editable annotations, trained weights, per-particle predictions, electrochemical raw data, complete DFT calculations, publisher materials, and dissertation submission files.

## Citation

For use of the repository code, cite the versioned repository release:

> Xu, G. *Oxide-Modified Pt Catalyst Dissertation Support*, version 1.0.0, 2026. https://github.com/iserlohn0522-cell/oxide-modified-pt-catalyst-dissertation-support

For scientific results or interpretations, cite the dissertation:

> Xu, G. *Interfacial Engineering of Metal Oxide-Modified Pt-Based Catalysts for Durable PEMFC Cathodes*. Ph.D. dissertation, University of Missouri-Columbia, 2026.

Machine-readable repository citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

Original code, tests, configurations, and repository-operational documentation are available under the MIT License. The exact scope and exclusions are stated in [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). External software and model names remain subject to their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
