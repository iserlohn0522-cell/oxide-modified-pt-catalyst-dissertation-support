# Chapter 5 TEM Computer-Vision Workflow

The released workflow separates five functions: particle detector training/evaluation, detector-box-prompted mask refinement, substrate segmentation, scale verification, and projected two-dimensional morphometry. Private images and identities are not required to inspect the interfaces.

## Particle dataset and detector

`manifests/ch5_ml_tem/source_manifest.template.csv` maps a public-safe `image_id` to a user-supplied image and LabelMe record. `split_manifest.template.yaml` assigns source images to user-selected partitions. Dataset windows that overlap an `ignore_region` are excluded by default. Particle boxes must be fully contained in an accepted window.

`build_detector_dataset.py` creates overlapping windows and records their source coordinates and rotations. These windows increase training exposure but are not independent scientific observations. `evaluate_source_images.py` returns rotation-zero predictions to source coordinates, applies non-maximum suppression, and performs one-to-one reference matching before reporting pooled and per-image metrics.

The production configuration is `configs/ch5_ml_tem/production_particle_model.yaml`. The matching sanitized checkpoint is a separate versioned release asset described in `models/particle_detector_model_card.md`.

`run_particle_inference.py` applies that detector to a directory of prepared windows and writes the JSON-lines format consumed by `evaluate_source_images.py`. Its default confidence is 0.25, matching the production configuration.

## Detector-box-prompted masks

In the dissertation workflow, detector boxes in source-image coordinates were supplied to SAM 3 through the Ultralytics interface to obtain particle masks. This repository documents that interface but does not redistribute Meta's provider checkpoint, the dissertation masks, or per-particle outputs. The reference-box versus detector-box aggregate mask-IoU comparison is included in `data/ch5_ml_tem/aggregate_results.csv`.

## Substrate consensus

The support footprint was segmented by three U-Net/ResNet-34 models. Each RGB image is resized to 768 × 768 by bilinear interpolation, divided by 255, passed through the model and sigmoid, thresholded at 0.3, 0.7, or 0.5 for the three respective seeds, restored to source dimensions by nearest-neighbor interpolation, and combined by a two-of-three vote.

The exact public configuration is `configs/ch5_ml_tem/production_substrate_consensus.yaml`. The three sanitized state dictionaries are separate release assets. Example command:

```text
python scripts/ch5_ml_tem/run_substrate_consensus.py \
  --image path/to/image.png \
  --weights seed20260623.pt seed20260624.pt seed20260625.pt \
  --output path/to/substrate_mask.png
```

## Scale verification

The workflow associated a detected scale-bar geometry with its text value and unit, followed by human checking. A record is eligible for physical-unit conversion only when `verification_status` is `accepted` or `verified`. The public schema is `configs/ch5_ml_tem/scale_verification_schema.yaml`; `pt_tem_cv.scale.calibrated_nm_per_pixel` enforces it.

## Projected morphometry

`pt_tem_cv.morphometry` calculates Euclidean closed-contour perimeter, projected area, equivalent-circle diameter, and circularity `4πA/P²` after applying the verified image scale. Circularity is clipped to [0, 1] for paper-facing output. These are descriptors of a three-dimensional particle's two-dimensional TEM projection.

## Public results and limits

`data/ch5_ml_tem/aggregate_results.csv` contains aggregate values reported in the dissertation, including the 11-image route-selection result and the 215-image deployment count. It contains no source-image identifiers, split membership, masks, or particle-level rows. The synthetic example under `examples/synthetic_tem/` validates software interfaces only and must not be read as a performance evaluation.
