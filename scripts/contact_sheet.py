"""Build class contact sheets from generated vial-position crops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(r"C:\Users\aamp7\Downloads\gen_v2_fisheye\out")
DEFAULT_OUTPUT = Path(r"C:\Users\aamp7\Downloads\gen_v2_fisheye\sheets")
CATEGORIES = (
    "pass",
    "missing_cap",
    "crooked_cap",
    "damaged_vial",
    "no_vial_not_yet_filled_true",
    "no_vial_not_yet_filled_false",
)
SCALE = 3
CAPTION_HEIGHT = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def ensure_external_output(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(f"Refusing to write contact sheets inside the repository: {path}")


def category_for(label: str, slot: str, not_yet_filled: set[str]) -> str:
    if label != "no_vial":
        return label
    suffix = "true" if slot in not_yet_filled else "false"
    return f"no_vial_not_yet_filled_{suffix}"


def make_tile(crop_path: Path) -> np.ndarray:
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        raise FileNotFoundError(f"Could not read crop: {crop_path}")
    enlarged = cv2.resize(
        crop,
        (crop.shape[1] * SCALE, crop.shape[0] * SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    tile = np.full(
        (enlarged.shape[0] + CAPTION_HEIGHT, enlarged.shape[1], 3),
        255,
        dtype=np.uint8,
    )
    tile[: enlarged.shape[0]] = enlarged
    cv2.putText(
        tile,
        crop_path.name,
        (5, enlarged.shape[0] + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return tile


def write_sheet(category: str, crop_paths: list[Path], output_path: Path) -> None:
    if not crop_paths:
        sheet = np.full((120, 700, 3), 255, dtype=np.uint8)
        cv2.putText(
            sheet,
            f"{category}: no samples",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        tiles = [make_tile(path) for path in crop_paths]
        columns = min(6, max(1, math.ceil(math.sqrt(len(tiles)))))
        rows = math.ceil(len(tiles) / columns)
        tile_height, tile_width = tiles[0].shape[:2]
        sheet = np.full(
            (rows * tile_height, columns * tile_width, 3), 230, dtype=np.uint8
        )
        for index, tile in enumerate(tiles):
            row, column = divmod(index, columns)
            y0 = row * tile_height
            x0 = column * tile_width
            sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"Could not write contact sheet: {output_path}")


def main() -> None:
    args = parse_args()
    ensure_external_output(args.out)
    document = json.loads((args.results / "labels.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[Path]] = {category: [] for category in CATEGORIES}

    for record in document["images"]:
        image_stem = Path(record["image"]).stem
        not_yet_filled = set(record["not_yet_filled"])
        for slot, label in record["labels"].items():
            category = category_for(label, slot, not_yet_filled)
            crop_path = args.results / "crops" / f"{image_stem}_{slot}.png"
            if not crop_path.is_file():
                raise FileNotFoundError(f"Missing crop: {crop_path}")
            grouped[category].append(crop_path)

    assigned = sum(len(paths) for paths in grouped.values())
    if assigned != 120:
        raise ValueError(f"Expected 120 classified crops, found {assigned}")

    args.out.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        output_path = args.out / f"{category}.png"
        write_sheet(category, grouped[category], output_path)
        print(f"{output_path} ({len(grouped[category])} crops)")


if __name__ == "__main__":
    main()
