# Substrate Consensus Model Card

## Intended use

The three `substrate_unet_resnet34_seed*.pt` assets are the production substrate-segmentation members used in the Chapter 5 morphometry workflow. Each asset is a PyTorch state dictionary for a U-Net with a ResNet-34 encoder. The three binary masks are combined by a two-of-three majority vote.

The model architecture, preprocessing, member thresholds, and consensus rule are recorded in `configs/ch5_ml_tem/production_substrate_consensus.yaml`. Public inference helpers are provided in `pt_tem_cv.substrate`.

## Provenance and verification

The weights were exported from the accepted checkpoints without optimizer state, local paths, training histories, or dataset identifiers. For every member, all exported state tensors exactly matched the accepted source model, and fixed-input forward comparisons gave a maximum absolute difference of 0.0. Asset digests are listed in `MODEL_REGISTRY.yaml`.

## Limitations

These models delineate the projected substrate region in TEM images. They do not infer a three-dimensional support geometry, material loading, or particle-support contact mechanism. Their outputs were used together with separately detected particle masks and human-verified image scale. The private source images, reference masks, and production outputs are not part of the release.

## License

The exported state dictionaries and original repository software are distributed under AGPL-3.0-or-later. PyTorch and `segmentation-models-pytorch` remain under their respective licenses; see `THIRD_PARTY_NOTICES.md`.
