"""Fit a monotonic OpenCV-fisheye camera to vial-slot correspondences."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fisheye import (
    MAX_PINHOLE_THETA,
    PRINCIPAL_POINT,
    corner_theta_max,
    distortion_derivative,
    distort_theta,
    minimum_derivative,
    project_world,
)


IMAGE_SIZE = (1920, 1080)
FOCAL_STARTS = (700.0, 900.0, 1100.0, 1400.0)
ZMODES = ("tube_min", "tube_max", "cap_center")
MIN_DERIVATIVE = 0.05
ANCHOR_PIXELS = {
    "G01": (1078.0, 406.0),
    "G05": (1067.0, 715.0),
    "G26": (750.0, 398.0),
    "G30": (744.0, 710.0),
    "S01": (1480.0, 630.0),
    "S05": (1466.0, 921.0),
}


@dataclass(frozen=True)
class FitResult:
    params: np.ndarray
    rms_px: float
    anchor_errors_px: np.ndarray
    loo_errors_px: np.ndarray
    loo_rms_px: float
    theta_max: float
    min_dtheta_d: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prims", type=Path, default=Path("calib/twin_prims.json")
    )
    parser.add_argument(
        "--init",
        type=Path,
        help="Optional fisheye camera JSON used as the pose initialization.",
    )
    parser.add_argument("--zmode", choices=ZMODES, default="tube_min")
    parser.add_argument(
        "--fit-json", type=Path, default=Path("calib/camera_fit.json")
    )
    return parser.parse_args()


def load_boxes(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    boxes = {}
    for prim_path, bounds in raw.items():
        box = np.asarray(bounds, dtype=np.float64)
        if box.shape != (2, 3) or not np.isfinite(box).all():
            raise ValueError(f"Invalid world box for {prim_path!r}")
        boxes[str(prim_path)] = box
    return boxes


def find_box(boxes: Mapping[str, np.ndarray], slot: str, part: str) -> np.ndarray:
    suffix = f"/{slot}_{part}"
    matches = [box for path, box in boxes.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one box ending in {suffix!r}; found {len(matches)}")
    return matches[0]


def slot_point(boxes: Mapping[str, np.ndarray], slot: str, zmode: str) -> np.ndarray:
    tube = find_box(boxes, slot, "tube")
    xy = tube.mean(axis=0)[:2]
    if zmode == "tube_min":
        z_value = tube[0, 2]
    elif zmode == "tube_max":
        z_value = tube[1, 2]
    elif zmode == "cap_center":
        cap_suffix = f"/{slot}_cap"
        caps = [box for path, box in boxes.items() if path.endswith(cap_suffix)]
        z_value = caps[0][:, 2].mean() if len(caps) == 1 else tube[1, 2]
    else:
        raise ValueError(f"Unknown zmode: {zmode}")
    return np.array([xy[0], xy[1], z_value], dtype=np.float64)


def anchor_arrays(
    boxes: Mapping[str, np.ndarray], zmode: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(ANCHOR_PIXELS)
    world = np.vstack([slot_point(boxes, name, zmode) for name in names])
    pixels = np.asarray([ANCHOR_PIXELS[name] for name in names], dtype=np.float64)
    return names, world, pixels


def default_initial_pose(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    center = points.mean(axis=0)
    position = np.array([center[0], center[1], center[2] + 1.2])
    rotation = np.diag([-1.0, 1.0, -1.0])
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return rotation_vector.ravel(), position, 0.0, 0.0


def load_initial_pose(
    init_path: Path | None, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if init_path is None:
        return default_initial_pose(points)
    raw = json.loads(init_path.read_text(encoding="utf-8"))
    rotation = np.asarray(raw["R_world_to_cv"], dtype=np.float64)
    position = np.asarray(raw["position_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("Initial rotation must be 3x3 and position must have 3 values")
    rotation_vector, _ = cv2.Rodrigues(rotation)
    if raw.get("model") == "opencv_fisheye":
        return (
            rotation_vector.ravel(),
            position,
            float(raw.get("k1", 0.0)),
            float(raw.get("k2", 0.0)),
        )
    return rotation_vector.ravel(), position, 0.0, 0.0


def unpack(params: np.ndarray):
    rotation, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    return (
        rotation,
        params[3:6],
        float(params[6]),
        float(params[7]),
        float(params[8]),
    )


def fitted_geometry(focal: float, k1: float, k2: float) -> tuple[float, float]:
    theta_max = corner_theta_max(IMAGE_SIZE, focal, k1, k2)
    return theta_max, minimum_derivative(k1, k2, theta_max)


def project(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    rotation, position, focal, k1, k2 = unpack(params)
    return project_world(points, rotation, position, focal, k1, k2)


def monotonic_penalty(params: np.ndarray) -> np.ndarray:
    focal, k1, k2 = (float(params[6]), float(params[7]), float(params[8]))
    corner_radius = max(
        math.hypot(u - PRINCIPAL_POINT[0], v - PRINCIPAL_POINT[1])
        for u, v in ((0.0, 0.0), (1919.0, 0.0), (0.0, 1079.0), (1919.0, 1079.0))
    )
    required_theta_d = corner_radius / focal
    coverage_violation = max(
        0.0, required_theta_d - float(distort_theta(MAX_PINHOLE_THETA, k1, k2))
    )
    try:
        theta_max = corner_theta_max(IMAGE_SIZE, focal, k1, k2)
    except ValueError:
        theta_max = MAX_PINHOLE_THETA
    theta_samples = np.linspace(0.0, theta_max, 65)
    derivative = distortion_derivative(theta_samples, k1, k2)
    derivative_violation = np.maximum(0.0, MIN_DERIVATIVE + 1.0e-4 - derivative)
    return np.concatenate(
        [100_000.0 * derivative_violation, [100_000.0 * coverage_violation]]
    )


def residuals(params: np.ndarray, world: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    try:
        reprojection = (project(world, params) - pixels).ravel()
    except ValueError:
        reprojection = np.full(pixels.size, 1.0e6, dtype=np.float64)
    return np.concatenate([reprojection, monotonic_penalty(params)])


def parameter_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    lower = np.array(
        [-2.0 * math.pi] * 3
        + [point_min[0] - 3.0, point_min[1] - 3.0, point_max[2] + 0.05]
        + [500.0, -1.0, -1.0]
    )
    upper = np.array(
        [2.0 * math.pi] * 3
        + [point_max[0] + 3.0, point_max[1] + 3.0, point_max[2] + 5.0]
        + [2500.0, 1.0, 1.0]
    )
    return lower, upper


def fit_multistart(
    world: np.ndarray,
    pixels: np.ndarray,
    initial_rotation: np.ndarray,
    initial_position: np.ndarray,
    initial_k1: float,
    initial_k2: float,
) -> np.ndarray:
    lower, upper = parameter_bounds(world)
    best = None
    for focal in FOCAL_STARTS:
        start = np.concatenate(
            [initial_rotation, initial_position, [focal, initial_k1, initial_k2]]
        )
        start = np.clip(start, lower + 1.0e-8, upper - 1.0e-8)
        result = least_squares(
            residuals,
            start,
            args=(world, pixels),
            bounds=(lower, upper),
            method="trf",
            x_scale=np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1000.0, 0.2, 0.2]),
            max_nfev=2_500,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        try:
            theta_max, min_derivative_value = fitted_geometry(
                float(result.x[6]), float(result.x[7]), float(result.x[8])
            )
        except ValueError:
            continue
        if min_derivative_value <= MIN_DERIVATIVE:
            continue
        pixel_residual = residuals(result.x, world, pixels)[: pixels.size]
        score = float(np.sum(pixel_residual * pixel_residual))
        if best is None or score < best[0]:
            best = (score, result.x, theta_max, min_derivative_value)
    if best is None:
        raise RuntimeError("No monotonic fisheye fit satisfied the field-of-view constraint")
    return best[1]


def evaluate_mode(
    boxes: Mapping[str, np.ndarray], zmode: str, init_path: Path | None
) -> tuple[list[str], FitResult]:
    names, world, pixels = anchor_arrays(boxes, zmode)
    initial_rotation, initial_position, initial_k1, initial_k2 = load_initial_pose(
        init_path, world
    )
    params = fit_multistart(
        world,
        pixels,
        initial_rotation,
        initial_position,
        initial_k1,
        initial_k2,
    )
    predicted = project(world, params)
    errors = np.linalg.norm(predicted - pixels, axis=1)
    rms = float(np.sqrt(np.mean(errors * errors)))

    loo_errors = []
    for held_out in range(len(names)):
        keep = np.arange(len(names)) != held_out
        loo_params = fit_multistart(
            world[keep],
            pixels[keep],
            params[:3],
            params[3:6],
            float(params[7]),
            float(params[8]),
        )
        prediction = project(world[held_out : held_out + 1], loo_params)[0]
        loo_errors.append(float(np.linalg.norm(prediction - pixels[held_out])))
    loo_array = np.asarray(loo_errors, dtype=np.float64)
    loo_rms = float(np.sqrt(np.mean(loo_array * loo_array)))
    theta_max, min_derivative_value = fitted_geometry(
        float(params[6]), float(params[7]), float(params[8])
    )
    return names, FitResult(
        params=params,
        rms_px=rms,
        anchor_errors_px=errors,
        loo_errors_px=loo_array,
        loo_rms_px=loo_rms,
        theta_max=theta_max,
        min_dtheta_d=min_derivative_value,
    )


def print_mode_details(names: Sequence[str], zmode: str, fit: FitResult) -> None:
    print(f"\n{zmode}")
    print("anchor  fit_error_px  loo_error_px")
    for name, error, loo_error in zip(
        names, fit.anchor_errors_px, fit.loo_errors_px, strict=True
    ):
        print(f"{name:>6}  {error:12.3f}  {loo_error:12.3f}")
    print(f"RMS: {fit.rms_px:.3f} px")
    print(f"Leave-one-out RMS: {fit.loo_rms_px:.3f} px")
    print(f"theta_max: {fit.theta_max:.6f} rad")
    print(f"minimum d(theta_d)/d(theta): {fit.min_dtheta_d:.6f}")


def save_fit(path: Path, zmode: str, fit: FitResult) -> None:
    rotation, position, focal, k1, k2 = unpack(fit.params)
    payload = {
        "model": "opencv_fisheye",
        "image_size": list(IMAGE_SIZE),
        "R_world_to_cv": rotation.tolist(),
        "position_m": position.tolist(),
        "focal_px": focal,
        "k1": k1,
        "k2": k2,
        "zmode": zmode,
        "rms_px": fit.rms_px,
        "loo_rms_px": fit.loo_rms_px,
        "theta_max": fit.theta_max,
        "min_dtheta_d": fit.min_dtheta_d,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    boxes = load_boxes(args.prims)
    evaluated = {}
    for zmode in ZMODES:
        names, fit = evaluate_mode(boxes, zmode, args.init)
        evaluated[zmode] = fit
        print_mode_details(names, zmode, fit)

    print("\nZ-mode comparison")
    print("zmode       rms_px  loo_rms_px  theta_max  min_derivative")
    for zmode in ZMODES:
        fit = evaluated[zmode]
        print(
            f"{zmode:<10}  {fit.rms_px:7.3f}  {fit.loo_rms_px:10.3f}  "
            f"{fit.theta_max:9.6f}  {fit.min_dtheta_d:14.6f}"
        )

    selected = evaluated[args.zmode]
    save_fit(args.fit_json, args.zmode, selected)
    print(f"\nSaved {args.zmode} fit to {args.fit_json}")


if __name__ == "__main__":
    main()
