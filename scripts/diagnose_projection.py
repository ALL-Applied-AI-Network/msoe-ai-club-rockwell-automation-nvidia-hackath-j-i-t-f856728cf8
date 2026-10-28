"""Diagnose fitted and generated slot projections without changing the generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fisheye import project_world


CORNER_ORDER = ("G26", "G01", "G30", "G05")
REAL_CORNERS = {
    "G26": (750.0, 398.0),
    "G01": (1078.0, 406.0),
    "G30": (744.0, 710.0),
    "G05": (1067.0, 715.0),
}
ZMODES = ("tube_min", "tube_max", "cap_center")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prims", type=Path, default=Path("calib/twin_prims.json")
    )
    parser.add_argument(
        "--camera", type=Path, default=Path("calib/camera_fit.json")
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            r"C:\Users\aamp7\Downloads\gen_v2_fisheye\out\labels.json"
        ),
    )
    return parser.parse_args()


def find_box(boxes: dict, slot: str, part: str) -> np.ndarray:
    suffix = f"/{slot}_{part}"
    matches = [value for path, value in boxes.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one box ending in {suffix!r}; found {len(matches)}")
    return np.asarray(matches[0], dtype=np.float64)


def slot_point(boxes: dict, slot: str, zmode: str) -> np.ndarray:
    tube = find_box(boxes, slot, "tube")
    xy = tube.mean(axis=0)[:2]
    if zmode == "tube_min":
        z_value = tube[0, 2]
    elif zmode == "tube_max":
        z_value = tube[1, 2]
    elif zmode == "cap_center":
        cap = find_box(boxes, slot, "cap")
        z_value = cap[:, 2].mean()
    else:
        raise ValueError(f"Unknown zmode: {zmode}")
    return np.array([xy[0], xy[1], z_value], dtype=np.float64)


def project_slots(
    boxes: dict, camera: dict, slots: list[str] | tuple[str, ...], zmode: str
) -> dict[str, np.ndarray]:
    world = np.vstack([slot_point(boxes, slot, zmode) for slot in slots])
    pixels = project_world(
        world,
        np.asarray(camera["R_world_to_cv"], dtype=np.float64),
        np.asarray(camera["position_m"], dtype=np.float64),
        float(camera["focal_px"]),
        float(camera["k1"]),
        float(camera["k2"]),
    )
    return {slot: pixel for slot, pixel in zip(slots, pixels, strict=True)}


def grid_size(points: dict[str, np.ndarray]) -> tuple[float, float]:
    width = (
        abs(float(points["G01"][0] - points["G26"][0]))
        + abs(float(points["G05"][0] - points["G30"][0]))
    ) / 2.0
    height = (
        abs(float(points["G30"][1] - points["G26"][1]))
        + abs(float(points["G05"][1] - points["G01"][1]))
    ) / 2.0
    return width, height


def main() -> None:
    args = parse_args()
    boxes = json.loads(args.prims.read_text(encoding="utf-8"))
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    real = {
        slot: np.asarray(REAL_CORNERS[slot], dtype=np.float64)
        for slot in CORNER_ORDER
    }

    print("REAL-CORNER COMPARISON")
    for zmode in ZMODES:
        projected = project_slots(boxes, camera, CORNER_ORDER, zmode)
        errors = []
        print(f"\n{zmode}")
        print("corner  projected_px        real_px             error_px")
        for slot in CORNER_ORDER:
            error = float(np.linalg.norm(projected[slot] - real[slot]))
            errors.append(error)
            print(
                f"{slot:>5}  ({projected[slot][0]:8.2f}, {projected[slot][1]:8.2f})  "
                f"({real[slot][0]:7.2f}, {real[slot][1]:7.2f})  {error:8.3f}"
            )
        rms = float(np.sqrt(np.mean(np.square(errors))))
        width, height = grid_size(projected)
        print(f"corner RMS: {rms:.3f} px")
        print(f"grid width: {width:.2f} px")
        print(f"grid height: {height:.2f} px")

    slots = [f"G{index:02d}" for index in range(1, 31)]
    label_positions = {
        slot: np.asarray(labels["positions"][slot], dtype=np.float64)
        for slot in slots
    }
    print("\nLABELS.JSON POSITION COMPARISON")
    matching_modes = []
    for zmode in ZMODES:
        projected = project_slots(boxes, camera, slots, zmode)
        errors = np.asarray(
            [np.linalg.norm(projected[slot] - label_positions[slot]) for slot in slots]
        )
        print(
            f"{zmode:<10} mean={errors.mean():8.3f} px  "
            f"rms={np.sqrt(np.mean(errors * errors)):8.3f} px  "
            f"max={errors.max():8.3f} px"
        )
        if float(errors.max()) <= 1.0:
            matching_modes.append(zmode)
    if matching_modes:
        print("labels.json matches within 1 px: " + ", ".join(matching_modes))
    else:
        print("labels.json matches none of the three zmodes within 1 px.")


if __name__ == "__main__":
    main()
