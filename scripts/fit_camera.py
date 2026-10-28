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
ZMODES = ("cap_min", "tube_max", "cap_center", "tube_min")
MIN_DERIVATIVE = 0.05
DISTORTION_BOUND = 0.5
BOUND_MARGIN = 0.01 * (2.0 * DISTORTION_BOUND)
GRID_CORNERS = ("G26", "G01", "G30", "G05")
ANCHOR_PIXELS = {
    "G01": (1078.0, 406.0),
    "G05": (1067.0, 715.0),
    "G26": (750.0, 398.0),
    "G30": (744.0, 710.0),
    "S01": (1480.0, 630.0),
    "S05": (1466.0, 921.0),
}
VIEW_122_PIXELS = {
    "G26": (748.0, 377.0),
    "G01": (1075.0, 387.0),
    "G30": (741.0, 694.0),
    "G05": (1065.0, 697.0),
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
    grid_width_px: float
    grid_height_px: float
    camera_height_m: float
    k1: float
    k2: float
    bound_hitting: bool
    plausible: bool


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
    parser.add_argument("--zmode", choices=("auto",) + ZMODES, default="auto")
    parser.add_argument(
        "--fit-json", type=Path, default=Path("calib/camera_fit.json")
    )
    parser.add_argument(
        "--fit-table", type=Path, default=Path("calib/fit_table.txt")
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
    elif zmode == "cap_min":
        cap_suffix = f"/{slot}_cap"
        caps = [box for path, box in boxes.items() if path.endswith(cap_suffix)]
        z_value = caps[0][0, 2] if len(caps) == 1 else tube[1, 2]
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
        + [500.0, -DISTORTION_BOUND, -DISTORTION_BOUND]
    )
    upper = np.array(
        [2.0 * math.pi] * 3
        + [point_max[0] + 3.0, point_max[1] + 3.0, point_max[2] + 5.0]
        + [2500.0, DISTORTION_BOUND, DISTORTION_BOUND]
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


def grid_size(projected: Mapping[str, np.ndarray]) -> tuple[float, float]:
    width = (
        abs(float(projected["G01"][0] - projected["G26"][0]))
        + abs(float(projected["G05"][0] - projected["G30"][0]))
    ) / 2.0
    height = (
        abs(float(projected["G30"][1] - projected["G26"][1]))
        + abs(float(projected["G05"][1] - projected["G01"][1]))
    ) / 2.0
    return width, height


def coefficient_hits_bound(value: float) -> bool:
    return min(
        abs(value + DISTORTION_BOUND), abs(value - DISTORTION_BOUND)
    ) <= BOUND_MARGIN


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
    corner_world = np.vstack(
        [slot_point(boxes, slot, zmode) for slot in GRID_CORNERS]
    )
    corner_pixels = project(corner_world, params)
    projected_corners = {
        slot: pixel
        for slot, pixel in zip(GRID_CORNERS, corner_pixels, strict=True)
    }
    grid_width, grid_height = grid_size(projected_corners)
    _, position, _, k1, k2 = unpack(params)
    camera_height = float(position[2] - world[:, 2].mean())
    bound_hitting = coefficient_hits_bound(k1) or coefficient_hits_bound(k2)
    plausible = (
        0.2 <= camera_height <= 3.0
        and theta_max < math.radians(80.0)
        and min_derivative_value > MIN_DERIVATIVE
    )
    return names, FitResult(
        params=params,
        rms_px=rms,
        anchor_errors_px=errors,
        loo_errors_px=loo_array,
        loo_rms_px=loo_rms,
        theta_max=theta_max,
        min_dtheta_d=min_derivative_value,
        grid_width_px=grid_width,
        grid_height_px=grid_height,
        camera_height_m=camera_height,
        k1=k1,
        k2=k2,
        bound_hitting=bound_hitting,
        plausible=plausible,
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
    print(f"k1: {fit.k1:.6f}; k2: {fit.k2:.6f}")
    print(f"k1/k2 within 1% of a bound: {'yes' if fit.bound_hitting else 'no'}")
    print(
        f"grid: {fit.grid_width_px:.2f} x {fit.grid_height_px:.2f} px; "
        f"camera height above anchor plane: {fit.camera_height_m:.4f} m"
    )


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


def comparison_table(evaluated: Mapping[str, FitResult]) -> str:
    lines = [
        "zmode       rms_px  loo_rms  theta_max  min_deriv       k1       k2  bound  plausible  grid_w  grid_h  cam_h_m"
    ]
    for zmode in ZMODES:
        fit = evaluated[zmode]
        lines.append(
            f"{zmode:<10}  {fit.rms_px:7.3f}  {fit.loo_rms_px:7.3f}  "
            f"{fit.theta_max:9.6f}  {fit.min_dtheta_d:9.6f}  "
            f"{fit.k1:7.4f}  {fit.k2:7.4f}  "
            f"{'yes' if fit.bound_hitting else 'no ':>5}  "
            f"{'yes' if fit.plausible else 'no ':>9}  "
            f"{fit.grid_width_px:6.2f}  {fit.grid_height_px:6.2f}  "
            f"{fit.camera_height_m:7.4f}"
        )
    return "\n".join(lines)


def validate_view_122(
    boxes: Mapping[str, np.ndarray], zmode: str, fit: FitResult
) -> str:
    world = np.vstack([slot_point(boxes, slot, zmode) for slot in GRID_CORNERS])
    projected_array = project(world, fit.params)
    projected = {
        slot: pixel
        for slot, pixel in zip(GRID_CORNERS, projected_array, strict=True)
    }
    real = {
        slot: np.asarray(VIEW_122_PIXELS[slot], dtype=np.float64)
        for slot in GRID_CORNERS
    }
    offsets = np.vstack([projected[slot] - real[slot] for slot in GRID_CORNERS])
    translation = offsets.mean(axis=0)
    residual_vectors = offsets - translation
    residuals_px = np.linalg.norm(residual_vectors, axis=1)
    projected_width, projected_height = grid_size(projected)
    real_width, real_height = grid_size(real)
    return "\n".join(
        [
            "view_122 translation-aligned validation (no refit)",
            f"removed translation: ({translation[0]:.3f}, {translation[1]:.3f}) px",
            f"scale ratio x: {projected_width / real_width:.6f}",
            f"scale ratio y: {projected_height / real_height:.6f}",
            f"mean residual: {residuals_px.mean():.3f} px",
            f"RMS residual: {np.sqrt(np.mean(residuals_px * residuals_px)):.3f} px",
            f"max residual: {residuals_px.max():.3f} px",
        ]
    )


def main() -> None:
    args = parse_args()
    boxes = load_boxes(args.prims)
    evaluated = {}
    for zmode in ZMODES:
        names, fit = evaluate_mode(boxes, zmode, args.init)
        evaluated[zmode] = fit
        print_mode_details(names, zmode, fit)

    table = comparison_table(evaluated)
    print("\nZ-mode comparison")
    print(table)

    if args.zmode == "auto":
        winner_mode = min(
            ZMODES,
            key=lambda mode: (
                evaluated[mode].bound_hitting,
                not evaluated[mode].plausible,
                evaluated[mode].loo_rms_px,
                evaluated[mode].rms_px,
            ),
        )
    else:
        winner_mode = args.zmode
    winner = evaluated[winner_mode]
    selection = (
        f"selected: {winner_mode} (bound_hitting={winner.bound_hitting}, "
        f"plausible={winner.plausible}, loo_rms_px={winner.loo_rms_px:.3f})"
    )
    validation = validate_view_122(boxes, winner_mode, winner)
    print("\n" + selection)
    print(validation)
    save_fit(args.fit_json, winner_mode, winner)
    args.fit_table.parent.mkdir(parents=True, exist_ok=True)
    args.fit_table.write_text(
        table + "\n\n" + selection + "\n" + validation + "\n",
        encoding="utf-8",
    )
    print(f"Saved winner to {args.fit_json}")
    print(f"Saved table to {args.fit_table}")


if __name__ == "__main__":
    main()
