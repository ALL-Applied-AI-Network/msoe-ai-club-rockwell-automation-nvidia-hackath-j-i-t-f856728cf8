"""Measure yellow-cap centering in generated pass crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_RESULTS = Path(r"C:\Users\aamp7\Downloads\gen_v2_fisheye\out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def largest_yellow_centroid(image: np.ndarray) -> tuple[float, float] | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([14, 65, 65], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if component_count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[largest, cv2.CC_STAT_AREA]) < 25:
        return None
    return float(centroids[largest, 0]), float(centroids[largest, 1])


def main() -> None:
    args = parse_args()
    document = json.loads((args.results / "labels.json").read_text(encoding="utf-8"))
    offsets = []
    crop_size = None
    missing = []

    for record in document["images"]:
        image_stem = Path(record["image"]).stem
        for slot, label in record["labels"].items():
            if label != "pass":
                continue
            crop_path = args.results / "crops" / f"{image_stem}_{slot}.png"
            crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if crop is None:
                raise FileNotFoundError(f"Could not read crop: {crop_path}")
            height, width = crop.shape[:2]
            if crop_size is None:
                crop_size = (width, height)
            elif crop_size != (width, height):
                raise ValueError("Pass crops do not all have the same size")
            centroid = largest_yellow_centroid(crop)
            if centroid is None:
                missing.append(crop_path.name)
                continue
            center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
            offsets.append(np.asarray(centroid) - center)

    if not offsets or crop_size is None:
        raise RuntimeError("No yellow cap centroids were found in pass crops")
    offset_array = np.vstack(offsets)
    mean = offset_array.mean(axis=0)
    std = offset_array.std(axis=0)
    current_shift = np.asarray(document.get("crop_shift", [8, -12]), dtype=np.float64)
    recommended_shift = current_shift + mean

    print(f"pass crops in labels: {len(offsets) + len(missing)}")
    print(f"yellow centroids found: {len(offsets)}")
    print(f"yellow centroids missing: {len(missing)}")
    print(f"crop size: {crop_size[0]} x {crop_size[1]} px")
    print(f"mean centroid offset x: {mean[0]:.3f} px")
    print(f"mean centroid offset y: {mean[1]:.3f} px")
    print(f"std centroid offset x:  {std[0]:.3f} px")
    print(f"std centroid offset y:  {std[1]:.3f} px")
    print(
        "crop-shift adjustment needed: "
        f"dx={mean[0]:+.3f} px, dy={mean[1]:+.3f} px"
    )
    print(
        "recommended crop shift: "
        f"dx={recommended_shift[0]:+.3f} px, dy={recommended_shift[1]:+.3f} px"
    )
    if missing:
        print("crops without a detected yellow centroid: " + ", ".join(missing))


if __name__ == "__main__":
    main()
