"""Draw slot centers, classes, and crop boxes on generated tray images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLASS_COLORS = {
    "pass": (0, 200, 0),
    "missing_cap": (0, 165, 255),
    "crooked_cap": (255, 80, 0),
    "damaged_vial": (180, 0, 255),
    "no_vial": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Generator results folder")
    parser.add_argument("--out", type=Path, required=True, help="External output folder")
    return parser.parse_args()


def ensure_external_output(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(f"Refusing to write overlay images inside the repository: {path}")


def main() -> None:
    args = parse_args()
    ensure_external_output(args.out)
    document = json.loads((args.results / "labels.json").read_text(encoding="utf-8"))
    crop_offsets = document.get("crop_offsets", [-38, -47, 42, 43])
    crop_shift = document.get("crop_shift", [8, -12])
    default_positions = document["positions"]
    args.out.mkdir(parents=True, exist_ok=True)

    for record in document["images"]:
        stem = Path(record["image"]).stem
        image_path = args.results / "images" / f"{stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        positions = record.get("projected_centers", default_positions)
        for slot, center_values in positions.items():
            center = (int(round(center_values[0])), int(round(center_values[1])))
            label = record["labels"][slot]
            color = CLASS_COLORS.get(label, (255, 255, 255))
            cv2.circle(image, center, 25, color, 2, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{slot} {label}",
                (center[0] - 36, center[1] - 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )

            shifted_x = int(round(center_values[0] + crop_shift[0]))
            shifted_y = int(round(center_values[1] + crop_shift[1]))
            top_left = (
                shifted_x + int(crop_offsets[0]),
                shifted_y + int(crop_offsets[1]),
            )
            bottom_right = (
                shifted_x + int(crop_offsets[2]),
                shifted_y + int(crop_offsets[3]),
            )
            cv2.rectangle(image, top_left, bottom_right, color, 1, cv2.LINE_AA)

        output_path = args.out / f"{stem}_overlay.png"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write overlay: {output_path}")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
