from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from pt_tem_cv.substrate import build_substrate_model, load_substrate_state_dict, majority_consensus, predict_substrate_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-member substrate segmentation consensus on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs=3, required=True)
    parser.add_argument("--thresholds", type=float, nargs=3, default=(0.3, 0.7, 0.5))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    masks = []
    for weight_path, threshold in zip(args.weights, args.thresholds, strict=True):
        model = build_substrate_model()
        model.load_state_dict(load_substrate_state_dict(weight_path, device=args.device))
        masks.append(predict_substrate_mask(image, model, threshold=threshold, device=args.device))
    consensus = majority_consensus(masks, required_votes=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(consensus, dtype=np.uint8) * 255, mode="L").save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
