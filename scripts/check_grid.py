"""Compare the synthetic projected G-grid with real view_001 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_LABELS = Path(
    r"C:\Users\aamp7\Downloads\gen_v2_fisheye\out\labels.json"
)
CORNER_ORDER = ("G26", "G01", "G30", "G05")
REAL_VIEW_001 = {
    "G26": (750.0, 398.0),
    "G01": (1078.0, 406.0),
    "G30": (744.0, 710.0),
    "G05": (1067.0, 715.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    return parser.parse_args()


def grid_size(points: dict[str, np.ndarray]) -> tuple[float, float]:
    top_width = abs(float(points["G01"][0] - points["G26"][0]))
    bottom_width = abs(float(points["G05"][0] - points["G30"][0]))
    left_height = abs(float(points["G30"][1] - points["G26"][1]))
    right_height = abs(float(points["G05"][1] - points["G01"][1]))
    return (top_width + bottom_width) / 2.0, (left_height + right_height) / 2.0


def main() -> None:
    args = parse_args()
    document = json.loads(args.labels.read_text(encoding="utf-8"))
    synthetic = {
        slot: np.asarray(document["positions"][slot], dtype=np.float64)
        for slot in CORNER_ORDER
    }
    real = {
        slot: np.asarray(REAL_VIEW_001[slot], dtype=np.float64)
        for slot in CORNER_ORDER
    }

    print("corner  synthetic_px       real_px            raw_delta_px")
    for slot in CORNER_ORDER:
        delta = synthetic[slot] - real[slot]
        print(
            f"{slot:>5}  ({synthetic[slot][0]:8.2f}, {synthetic[slot][1]:8.2f})  "
            f"({real[slot][0]:7.2f}, {real[slot][1]:7.2f})  "
            f"({delta[0]:8.2f}, {delta[1]:8.2f})"
        )

    synthetic_width, synthetic_height = grid_size(synthetic)
    real_width, real_height = grid_size(real)
    scale_x = synthetic_width / real_width
    scale_y = synthetic_height / real_height

    raw_offsets = np.vstack(
        [synthetic[slot] - real[slot] for slot in CORNER_ORDER]
    )
    translation = raw_offsets.mean(axis=0)
    aligned_offsets = raw_offsets - translation
    aligned_distances = np.linalg.norm(aligned_offsets, axis=1)

    print()
    print(f"synthetic grid width:  {synthetic_width:.2f} px")
    print(f"synthetic grid height: {synthetic_height:.2f} px")
    print(f"real grid width:       {real_width:.2f} px")
    print(f"real grid height:      {real_height:.2f} px")
    print(f"scale ratio x (synthetic/real): {scale_x:.4f}")
    print(f"scale ratio y (synthetic/real): {scale_y:.4f}")
    print(
        "removed translation: "
        f"({translation[0]:.2f}, {translation[1]:.2f}) px"
    )
    print(f"mean corner offset after translation: {aligned_distances.mean():.2f} px")
    print(f"max corner offset after translation:  {aligned_distances.max():.2f} px")

    average_scale = (scale_x + scale_y) / 2.0
    if average_scale > 1.02:
        verdict = "YES: the synthetic render is zoomed in relative to the real view."
    elif average_scale < 0.98:
        verdict = "NO: the synthetic render is zoomed out relative to the real view."
    else:
        verdict = "NO: the synthetic and real views have approximately the same zoom."
    print(verdict)


if __name__ == "__main__":
    main()
