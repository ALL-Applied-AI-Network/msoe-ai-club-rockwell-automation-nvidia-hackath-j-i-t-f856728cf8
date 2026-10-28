"""Compare defect crops numerically with the median generated pass crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


TARGET_CLASSES = ("damaged_vial", "crooked_cap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Generator output folder")
    return parser.parse_args()


def load_crop(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read crop: {path}")
    return image.astype(np.float32)


def main() -> None:
    args = parse_args()
    document = json.loads((args.out / "labels.json").read_text(encoding="utf-8"))
    by_class: dict[str, list[tuple[str, Path]]] = {
        "pass": [],
        **{label: [] for label in TARGET_CLASSES},
    }
    for record in document["images"]:
        image_stem = Path(record["image"]).stem
        for slot, label in record["labels"].items():
            if label not in by_class:
                continue
            crop_path = args.out / "crops" / f"{image_stem}_{slot}.png"
            by_class[label].append((crop_path.name, crop_path))

    if not by_class["pass"]:
        raise RuntimeError("No pass crops are available for the median reference")
    pass_images = [load_crop(path) for _, path in by_class["pass"]]
    shapes = {image.shape for image in pass_images}
    if len(shapes) != 1:
        raise ValueError("Pass crops do not all have the same dimensions")
    median_pass = np.median(np.stack(pass_images, axis=0), axis=0)

    print(f"pass reference crops: {len(pass_images)}")
    print(f"crop shape: {pass_images[0].shape[1]} x {pass_images[0].shape[0]} px")
    for label in TARGET_CLASSES:
        differences = []
        print(f"\n{label}")
        for filename, path in by_class[label]:
            crop = load_crop(path)
            if crop.shape != median_pass.shape:
                raise ValueError(f"Unexpected crop shape: {path}")
            difference = float(np.mean(np.abs(crop - median_pass)))
            differences.append(difference)
            print(f"{filename}: {difference:.3f}")
        if differences:
            print(f"{label} mean: {np.mean(differences):.3f}")
        else:
            print(f"{label} mean: no samples")


if __name__ == "__main__":
    main()
