from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the released Ultralytics particle detector on prepared image windows.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Particle inference requires: pip install 'pt-tem-cv[ml]'") from exc

    paths = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    detector = YOLO(str(args.weights))
    results = detector.predict(
        source=[str(path) for path in paths],
        conf=args.confidence,
        iou=args.iou,
        imgsz=args.image_size,
        device=args.device,
        verbose=False,
        stream=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for result in results:
            boxes = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                boxes = [{"box": box.tolist(), "score": float(score)} for box, score in zip(xyxy, scores, strict=True)]
            handle.write(json.dumps({"patch_name": Path(result.path).stem, "predictions": boxes}) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
