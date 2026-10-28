"""Build one twelve-crop inspection sheet per damaged-vial type."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DAMAGE_TYPES = ("crush", "bend", "lean_crush", "broken_top")
TILE_SCALE = 3
CAPTION_HEIGHT = 34
GRID_COLUMNS = 4
GRID_ROWS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Generator output folder")
    parser.add_argument("--out", type=Path, required=True, help="External output folder")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Labels document; defaults to the sole *labels.json in results.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def ensure_external_output(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(f"Refusing to write damage sheets inside the repository: {path}")


def resolve_labels(results: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(results.glob("*labels.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one *labels.json in {results}, found {len(candidates)}; "
            "pass --labels explicitly"
        )
    return candidates[0]


def make_tile(crop_path: Path, damage_type: str, strength: float) -> np.ndarray:
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        raise FileNotFoundError(f"Could not read crop: {crop_path}")
    enlarged = cv2.resize(
        crop,
        (crop.shape[1] * TILE_SCALE, crop.shape[0] * TILE_SCALE),
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
        f"{damage_type} {strength:.4g}",
        (4, enlarged.shape[0] + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    args = parse_args()
    ensure_external_output(args.out)
    labels_path = resolve_labels(args.results, args.labels)
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[Path, float]]] = {
        damage_type: [] for damage_type in DAMAGE_TYPES
    }
    for record in document["images"]:
        image_stem = Path(record["image"]).stem
        for slot, label in record["labels"].items():
            if label != "damaged_vial":
                continue
            damage = record["variations"][slot]["damage"]
            damage_type = damage["kind"]
            if damage_type not in grouped:
                continue
            grouped[damage_type].append(
                (
                    args.results / "crops" / f"{image_stem}_{slot}.png",
                    float(damage["strength"]),
                )
            )

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for damage_type in DAMAGE_TYPES:
        samples = grouped[damage_type]
        selected = (
            rng.sample(samples, 12) if len(samples) > 12 else list(samples)
        )
        tiles = [
            make_tile(crop_path, damage_type, strength)
            for crop_path, strength in selected
        ]
        if tiles:
            tile_height = max(tile.shape[0] for tile in tiles)
            tile_width = max(tile.shape[1] for tile in tiles)
        else:
            tile_height, tile_width = 90 * TILE_SCALE + CAPTION_HEIGHT, 80 * TILE_SCALE
        sheet = np.full(
            (GRID_ROWS * tile_height, GRID_COLUMNS * tile_width, 3),
            230,
            dtype=np.uint8,
        )
        for index, tile in enumerate(tiles):
            row, column = divmod(index, GRID_COLUMNS)
            height, width = tile.shape[:2]
            y0 = row * tile_height + (tile_height - height) // 2
            x0 = column * tile_width + (tile_width - width) // 2
            sheet[y0 : y0 + height, x0 : x0 + width] = tile
        output_path = args.out / f"{damage_type}.png"
        if not cv2.imwrite(str(output_path), sheet):
            raise OSError(f"Could not write damage sheet: {output_path}")
        print(f"{output_path} ({len(selected)} of 12 tiles populated)")


if __name__ == "__main__":
    main()
