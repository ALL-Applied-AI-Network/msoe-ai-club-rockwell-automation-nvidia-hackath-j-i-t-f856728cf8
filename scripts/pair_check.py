"""Compare each crop with its closest of five randomly selected pass crops."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Generator output folder")
    parser.add_argument(
        "--pass-samples",
        type=int,
        default=5,
        help="Number of random pass references considered per crop.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--labels",
        type=Path,
        help="Labels document; defaults to the sole *labels.json in out.",
    )
    return parser.parse_args()


def load_crop(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read crop: {path}")
    return image.astype(np.float32)


def resolve_labels(out: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(out.glob("*labels.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one *labels.json in {out}, found {len(candidates)}; "
            "pass --labels explicitly"
        )
    return candidates[0]


def main() -> None:
    args = parse_args()
    if args.pass_samples < 1:
        raise ValueError("--pass-samples must be at least 1")
    labels_path = resolve_labels(args.out, args.labels)
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    by_class: dict[str, list[tuple[str, Path, dict, str]]] = {
        label: [] for label in CLASSES
    }
    class_counts: Counter[str] = Counter()
    for record in document["images"]:
        image_stem = Path(record["image"]).stem
        for slot, label in record["labels"].items():
            class_counts[label] += 1
            crop_path = args.out / "crops" / f"{image_stem}_{slot}.png"
            by_class[label].append((crop_path.name, crop_path, record, slot))

    if len(by_class["pass"]) < 2:
        raise RuntimeError("At least two pass crops are required")
    loaded: dict[Path, np.ndarray] = {}

    def image_for(path: Path) -> np.ndarray:
        if path not in loaded:
            loaded[path] = load_crop(path)
        return loaded[path]

    pass_shape = image_for(by_class["pass"][0][1]).shape
    rng = random.Random(args.seed)

    def closest_pass_difference(path: Path) -> float:
        eligible = [item for item in by_class["pass"] if item[1] != path]
        references = rng.sample(
            eligible, min(args.pass_samples, len(eligible))
        )
        crop = image_for(path)
        if crop.shape != pass_shape:
            raise ValueError(f"Unexpected crop shape: {path}")
        return min(
            float(np.mean(np.abs(crop - image_for(reference_path))))
            for _, reference_path, _, _ in references
        )

    print(
        f"image size: {document['image_size'][0]} x {document['image_size'][1]} px"
    )
    print(f"crop size: {pass_shape[1]} x {pass_shape[0]} px")
    print("label counts:")
    for label in CLASSES:
        print(f"  {label}: {class_counts[label]}")
    print(
        f"metric: per-crop minimum mean absolute difference to "
        f"{args.pass_samples} random pass crops"
    )

    for label in CLASSES:
        differences = []
        print(f"\n{label}")
        for filename, path, record, slot in by_class[label]:
            difference = closest_pass_difference(path)
            differences.append(difference)
            if label == "damaged_vial":
                damage = record["variations"][slot].get("damage", {})
                print(
                    f"{filename}: {difference:.3f}  "
                    f"type={damage.get('kind', 'unknown')}  "
                    f"strength={damage.get('strength', 'unknown')}"
                )
        if differences:
            print(f"{label} mean minimum difference: {np.mean(differences):.3f}")
        else:
            print(f"{label} mean minimum difference: no samples")


if __name__ == "__main__":
    main()
