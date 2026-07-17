# Chapter 5 Detector-Side Workflow

The released Chapter 5 code covers the detector-side portion of the TEM particle workflow: source-image records, overlapping-window dataset construction, rotation and controlled train-time transformations, detector screening and training, and source-image-level box evaluation.

## User-supplied inputs

The repository includes schema templates, not dissertation image identities.

- `manifests/ch5_ml_tem/source_manifest.template.csv` maps an `image_id` to an image and editable LabelMe JSON record.
- `manifests/ch5_ml_tem/split_manifest.template.yaml` defines user-selected train, validation, and test membership by source image.
- `configs/ch5_ml_tem/dataset_template.yaml` defines window size, stride, rotations, labels, and ignore-region handling.

The expected particle label is `Pt_NPs`. Regions labeled `ignore_region` are excluded by default: dataset windows overlapping an ignore region are not materialized, and source-image evaluation excludes ground-truth and predicted boxes whose centers fall inside such a region. The summary records the number of excluded regions, references, and predictions.

## Build a sliding-window dataset

```text
python scripts/ch5_ml_tem/build_detector_dataset.py \
  --config configs/ch5_ml_tem/dataset_template.yaml \
  --source-manifest path/to/source_manifest.csv \
  --split-manifest path/to/split_manifest.yaml \
  --output-root path/to/output
```

Boxes are included only when the complete box lies within a window. Tile and rotation identifiers are recorded in `patch_manifest.csv`; these computational records do not represent additional TEM fields.

## Train or inspect screens

After installing the `ml` optional dependencies, one detector can be trained with `scripts/ch5_ml_tem/train_detector.py`. The family, augmentation, and factorial-screen scripts accept `--list-only` so their planned runs can be inspected without training.

The configuration files preserve the tested model/augmentation structures without publishing the dissertation's images, split identities, checkpoints, or results.

## Source-image evaluation

`scripts/ch5_ml_tem/evaluate_source_images.py` expects one JSON object per patch line. Each line contains a `patch_name` and local `xyxy` predictions with confidence scores, for example:

```json
{"patch_name":"sample_001__r000__x00000_y00000","predictions":[{"box":[20,30,45,55],"score":0.87}]}
```

Rotation-zero window predictions are translated back to source-image coordinates, merged by non-maximum suppression, and matched one-to-one to the reference boxes at the requested IoU threshold. The output contains pooled and per-image precision, recall, F1, and count-error summaries.

The dissertation's subsequent mask-refinement step used detector boxes as prompts to SAM 3 through the Ultralytics interface. Production images, masks, and model weights are outside this public package.
