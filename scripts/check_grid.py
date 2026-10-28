"""Compare the synthetic projected G-grid with real view_001 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_RESULTS = Path(r"C:\Users\aamp7\Downloads\gen_v2_fisheye\out")
DEFAULT_CAMERA = Path("calib/camera_fit.json")
CORNER_ORDER = ("G26", "G01", "G30", "G05")
REAL_VIEW_001 = {
    "G26": (750.0, 398.0),
    "G01": (1078.0, 406.0),
    "G30": (744.0, 710.0),
    "G05": (1067.0, 715.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    return parser.parse_args()


def grid_size(points: dict[str, np.ndarray]) -> tuple[float, float]:
    top_width = abs(float(points["G01"][0] - points["G26"][0]))
    bottom_width = abs(float(points["G05"][0] - points["G30"][0]))
    left_height = abs(float(points["G30"][1] - points["G26"][1]))
    right_height = abs(float(points["G05"][1] - points["G01"][1]))
    return (top_width + bottom_width) / 2.0, (left_height + right_height) / 2.0


def main() -> None:
    args = parse_args()
    document = json.loads(
        (args.results / "labels.json").read_text(encoding="utf-8")
    )
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    real = {
        slot: np.asarray(REAL_VIEW_001[slot], dtype=np.float64)
        for slot in CORNER_ORDER
    }
    real_width, real_height = grid_size(real)
    print(f"real grid width:       {real_width:.2f} px")
    print(f"real grid height:      {real_height:.2f} px")
    print()
    print("PER-IMAGE JITTERED GRID SCALE")
    print(
        "image      grid_w  grid_h  scale_x  scale_y  "
        "shift_px(x,y)       rotation_deg(x,y,z)       translation_world_m"
    )
    scale_rows = []
    for record in document["images"]:
        positions = {
            slot: np.asarray(record["projected_centers"][slot], dtype=np.float64)
            for slot in CORNER_ORDER
        }
        width, height = grid_size(positions)
        scale_x = width / real_width
        scale_y = height / real_height
        scale_rows.append((scale_x, scale_y))
        jitter = record["camera_jitter"]
        shift = jitter["requested_shift_px"]
        rotation = jitter["rotation_xyz_deg"]
        translation = jitter["translation_world_m"]
        print(
            f"{record['image']:<9}  {width:6.2f}  {height:6.2f}  "
            f"{scale_x:7.4f}  {scale_y:7.4f}  "
            f"({shift[0]:+6.2f},{shift[1]:+6.2f})  "
            f"({rotation[0]:+6.3f},{rotation[1]:+6.3f},{rotation[2]:+6.3f})  "
            f"({translation[0]:+.5f},{translation[1]:+.5f},{translation[2]:+.5f})"
        )

    scales = np.asarray(scale_rows, dtype=np.float64)
    mean_scale = scales.mean(axis=0)
    print()
    print(f"mean scale ratio x: {mean_scale[0]:.4f}")
    print(f"mean scale ratio y: {mean_scale[1]:.4f}")
    print(f"scale ratio x range: {scales[:, 0].min():.4f} to {scales[:, 0].max():.4f}")
    print(f"scale ratio y range: {scales[:, 1].min():.4f} to {scales[:, 1].max():.4f}")

    average_scale = float(mean_scale.mean())
    if average_scale > 1.02:
        verdict = "YES: the synthetic render is zoomed in relative to the real view."
    elif average_scale < 0.98:
        verdict = "NO: the synthetic render is zoomed out relative to the real view."
    else:
        verdict = "NO: the synthetic and real views have approximately the same zoom."
    print(verdict)

    focal = float(camera["focal_px"])
    lower, upper = 500.0, 2500.0
    one_percent_span = 0.01 * (upper - lower)
    distance_to_bound = min(focal - lower, upper - focal)
    within = distance_to_bound <= one_percent_span
    print()
    print(f"focal_px: {focal:.3f} px")
    print(f"focal bounds: [{lower:.0f}, {upper:.0f}] px")
    print(f"distance to nearest bound: {distance_to_bound:.3f} px")
    print(f"within 1% of a focal bound: {'yes' if within else 'no'}")


if __name__ == "__main__":
    main()
