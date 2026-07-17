# Third-Party Notices

This repository does not vendor third-party source code or provider checkpoints. Installing dependencies or obtaining provider weights is a separate action governed by each provider's current terms.

The core package uses NumPy, Pillow, PyYAML, and Matplotlib. Optional detector and substrate workflows use PyTorch, Ultralytics, and `segmentation-models-pytorch`. Their package metadata and upstream repositories identify their respective licenses; users should retain the notices installed with those packages.

The released particle detector uses the Ultralytics YOLO architecture and was verified with Ultralytics 8.4.9. Ultralytics identifies its open-source software and trained-model route as AGPL-3.0 unless a separate enterprise license applies. See https://www.ultralytics.com/license and https://www.ultralytics.com/legal/agpl-3-0-software-license.

The dissertation workflow also used detector-box prompts with Meta SAM 3. This repository does not redistribute a SAM 3 checkpoint or Meta source code. The official SAM 3 repository and model page associate it with the custom SAM License; obtain it only from the provider and review the current terms: https://github.com/facebookresearch/sam3 and https://huggingface.co/facebook/sam3.

Additional model-specific scope is recorded in `models/third_party_models.md`. Provider terms were checked on 2026-07-17; the linked current terms control over this summary.
