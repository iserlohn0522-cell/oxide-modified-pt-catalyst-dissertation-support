# Particle Detector Model Card

## Intended use

`particle_detector_yolov8m_seg_dissertation_v1.pt` is the production YOLOv8m-seg checkpoint used to localize Pt nanoparticles in the Chapter 5 TEM workflow. It supports inspection and application of the released method to user-supplied images; it does not include the dissertation image set, annotations, split membership, or per-particle outputs.

The model was trained on overlapping 512-pixel windows. Predictions used for scientific evaluation were returned to the source-image coordinates and merged before scoring. Tiles were computational units, whereas source TEM images and material states remained the measurement units.

## Configuration and provenance

The public training parameters are recorded in `configs/ch5_ml_tem/production_particle_model.yaml`. The sanitized checkpoint contains the inference model and public-safe runtime metadata. Its parameters are tensor-identical to the accepted source checkpoint, and a fixed-input forward comparison gave a maximum absolute difference of 0.0. The checkpoint-embedded 100-epoch training curve was also matched exactly to the accepted training record before sanitization.

The release uses the Ultralytics checkpoint format for ecosystem compatibility. This is a pickle-based format; do not load an untrusted or hash-mismatched file. The sanitization/equivalence environment used Ultralytics 8.4.9 and PyTorch 2.12.1+cpu.

The SHA-256 digest and byte count are in `MODEL_REGISTRY.yaml`. Verify the downloaded asset before use.

## Limitations

Performance values in the dissertation describe the frozen dissertation evaluation set and should not be assumed for new microscopes, magnifications, contrast conditions, materials, or annotation policies. Human review remains appropriate for scientific use. The checkpoint does not determine physical scale and does not by itself convert detections into calibrated morphometry.

## License

The release checkpoint and original repository software are distributed under AGPL-3.0-or-later. Ultralytics software and base-model material remain subject to Ultralytics' licensing terms; see `third_party_models.md` and `THIRD_PARTY_NOTICES.md`.
