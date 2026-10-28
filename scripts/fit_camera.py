"""Fit a pinhole camera with radial distortion to vial-slot correspondences."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


IMAGE_SIZE = (1920, 1080)
PRINCIPAL_POINT = np.array([960.0, 540.0], dtype=np.float64)
FOCAL_STARTS = (700.0, 900.0, 1100.0, 1400.0)
ZMODES = ("tube_min", "tube_max", "cap_center")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prims",
        type=Path,
        default=Path("calib/twin_prims.json"),
        help="Twin world-axis bounding boxes.",
    )
    parser.add_argument(
        "--init",
        type=Path,
        help="Optional CapSealCam.json with R_world_to_cv, position_m, and focal_px_final.",
    )
    parser.add_argument("--zmode", choices=ZMODES, default="tube_min")
    parser.add_argument(
        "--fit-json",
        type=Path,
        default=Path("calib/camera_fit.json"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(r"C:\Users\aamp7\Downloads\real_examples\tray_000001.jpg"),
    )
    parser.add_argument(
        "--check-image",
        type=Path,
        default=Path(r"C:\Users\aamp7\Downloads\fit_check\tray_000001_fit.png"),
    )
    return parser.parse_args()


def load_boxes(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    boxes: dict[str, np.ndarray] = {}
    for prim_path, bounds in raw.items():
        box = np.asarray(bounds, dtype=np.float64)
        if box.shape != (2, 3) or not np.isfinite(box).all():
            raise ValueError(f"Invalid world box for {prim_path!r}")
        boxes[str(prim_path)] = box
    return boxes


def find_box(boxes: Mapping[str, np.ndarray], slot: str, kind: str) -> np.ndarray:
    suffix = f"/{slot}_{kind}"
    matches = [box for path, box in boxes.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one box ending in {suffix!r}; found {len(matches)}")
    return matches[0]


def slot_point(
    boxes: Mapping[str, np.ndarray], slot: str, zmode: str
) -> np.ndarray:
    tube = find_box(boxes, slot, "tube")
    xy = tube.mean(axis=0)[:2]
    if zmode == "tube_min":
        z = tube[0, 2]
    elif zmode == "tube_max":
        z = tube[1, 2]
    elif zmode == "cap_center":
        cap_suffix = f"/{slot}_cap"
        caps = [box for path, box in boxes.items() if path.endswith(cap_suffix)]
        # Scrap positions can intentionally have no cap primitive. In that case,
        # the tube top is the closest defined surface for this height choice.
        z = caps[0][:, 2].mean() if len(caps) == 1 else tube[1, 2]
    else:
        raise ValueError(f"Unknown zmode: {zmode}")
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def anchor_arrays(
    boxes: Mapping[str, np.ndarray], zmode: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(ANCHOR_PIXELS)
    world = np.vstack([slot_point(boxes, name, zmode) for name in names])
    pixels = np.asarray([ANCHOR_PIXELS[name] for name in names], dtype=np.float64)
    return names, world, pixels


def default_initial_pose(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    position = np.array([center[0], center[1], center[2] + 1.2])
    rotation = np.diag([-1.0, 1.0, -1.0])
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return rotation_vector.ravel(), position, 0.0


def load_initial_pose(
    init_path: Path | None, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    if init_path is None:
        return default_initial_pose(points)

    raw = json.loads(init_path.read_text(encoding="utf-8"))
    rotation = np.asarray(raw["R_world_to_cv"], dtype=np.float64)
    position = np.asarray(raw["position_m"], dtype=np.float64)
    if rotation.shape != (3, 3) or position.shape != (3,):
        raise ValueError("Initial rotation must be 3x3 and position must have 3 values")
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return rotation_vector.ravel(), position, float(raw.get("k1", 0.0))


def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    rotation, _ = cv2.Rodrigues(params[:3].reshape(3, 1))
    return rotation, params[3:6], float(params[6]), float(params[7])


def project(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    rotation, position, focal, k1 = unpack(params)
    camera_points = (rotation @ (points - position).T).T
    depth = camera_points[:, 2]
    if np.any(depth <= 1.0e-8):
        raise ValueError("One or more points project behind the camera")
    normalized = camera_points[:, :2] / depth[:, None]
    radius_squared = np.sum(normalized * normalized, axis=1)
    distortion = 1.0 + k1 * radius_squared
    return focal * normalized * distortion[:, None] + PRINCIPAL_POINT


def residuals(params: np.ndarray, world: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    try:
        predicted = project(world, params)
    except ValueError:
        return np.full(pixels.size, 1.0e6, dtype=np.float64)
    return (predicted - pixels).ravel()


def parameter_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    lower = np.array(
        [-2.0 * math.pi] * 3
        + [point_min[0] - 3.0, point_min[1] - 3.0, point_max[2] + 0.05]
        + [200.0, -2.0]
    )
    upper = np.array(
        [2.0 * math.pi] * 3
        + [point_max[0] + 3.0, point_max[1] + 3.0, point_max[2] + 5.0]
        + [5000.0, 2.0]
    )
    return lower, upper


def fit_multistart(
    world: np.ndarray,
    pixels: np.ndarray,
    initial_rotation: np.ndarray,
    initial_position: np.ndarray,
    initial_k1: float,
) -> np.ndarray:
    lower, upper = parameter_bounds(world)
    best = None
    for focal in FOCAL_STARTS:
        start = np.concatenate(
            [initial_rotation, initial_position, [focal, initial_k1]]
        )
        start = np.clip(start, lower + 1.0e-9, upper - 1.0e-9)
        result = least_squares(
            residuals,
            start,
            args=(world, pixels),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=20_000,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        score = float(np.sum(result.fun * result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    if best is None:
        raise RuntimeError("Camera fit produced no result")
    return best[1]


def evaluate_mode(
    boxes: Mapping[str, np.ndarray], zmode: str, init_path: Path | None
) -> tuple[list[str], np.ndarray, np.ndarray, FitResult]:
    names, world, pixels = anchor_arrays(boxes, zmode)
    initial_rotation, initial_position, initial_k1 = load_initial_pose(init_path, world)
    params = fit_multistart(
        world, pixels, initial_rotation, initial_position, initial_k1
    )
    predicted = project(world, params)
    errors = np.linalg.norm(predicted - pixels, axis=1)
    rms = float(np.sqrt(np.mean(errors * errors)))

    loo_errors = []
    full_rotation = params[:3]
    full_position = params[3:6]
    full_k1 = float(params[7])
    for held_out in range(len(names)):
        keep = np.arange(len(names)) != held_out
        loo_params = fit_multistart(
            world[keep],
            pixels[keep],
            full_rotation,
            full_position,
            full_k1,
        )
        prediction = project(world[held_out : held_out + 1], loo_params)[0]
        loo_errors.append(float(np.linalg.norm(prediction - pixels[held_out])))
    loo_array = np.asarray(loo_errors, dtype=np.float64)
    loo_rms = float(np.sqrt(np.mean(loo_array * loo_array)))
    return names, world, pixels, FitResult(params, rms, errors, loo_array, loo_rms)


def print_mode_details(names: Sequence[str], zmode: str, fit: FitResult) -> None:
    print(f"\n{zmode}")
    print("anchor  fit_error_px  loo_error_px")
    for name, error, loo_error in zip(
        names, fit.anchor_errors_px, fit.loo_errors_px, strict=True
    ):
        print(f"{name:>6}  {error:12.3f}  {loo_error:12.3f}")
    print(f"RMS: {fit.rms_px:.3f} px")
    print(f"Leave-one-out RMS: {fit.loo_rms_px:.3f} px")


def save_fit(path: Path, zmode: str, fit: FitResult) -> None:
    rotation, position, focal, k1 = unpack(fit.params)
    payload = {
        "image_size": list(IMAGE_SIZE),
        "R_world_to_cv": rotation.tolist(),
        "position_m": position.tolist(),
        "focal_px": focal,
        "k1": k1,
        "zmode": zmode,
        "rms_px": fit.rms_px,
        "loo_rms_px": fit.loo_rms_px,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_external_image_path(path: Path) -> None:
    repository = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(repository)
    except ValueError:
        return
    raise ValueError(f"Refusing to write a generated image inside the repository: {path}")


def draw_check_image(
    boxes: Mapping[str, np.ndarray],
    zmode: str,
    fit: FitResult,
    input_path: Path,
    output_path: Path,
) -> None:
    ensure_external_image_path(output_path)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    if (image.shape[1], image.shape[0]) != IMAGE_SIZE:
        raise ValueError(
            f"Expected image size {IMAGE_SIZE}; got {(image.shape[1], image.shape[0])}"
        )

    names = [f"G{index:02d}" for index in range(1, 31)]
    world = np.vstack([slot_point(boxes, name, zmode) for name in names])
    pixels = project(world, fit.params)
    for index, pixel in enumerate(pixels, start=1):
        center = tuple(np.rint(pixel).astype(int))
        cv2.circle(image, center, 25, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (center[0] - 13, center[1] + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not write image: {output_path}")


def main() -> None:
    args = parse_args()
    boxes = load_boxes(args.prims)
    evaluated: dict[str, FitResult] = {}
    names_by_mode: dict[str, list[str]] = {}

    for zmode in ZMODES:
        names, _, _, fit = evaluate_mode(boxes, zmode, args.init)
        names_by_mode[zmode] = names
        evaluated[zmode] = fit
        print_mode_details(names, zmode, fit)

    print("\nZ-mode comparison")
    print("zmode       rms_px  loo_rms_px")
    for zmode in ZMODES:
        fit = evaluated[zmode]
        print(f"{zmode:<10}  {fit.rms_px:7.3f}  {fit.loo_rms_px:10.3f}")

    selected = evaluated[args.zmode]
    save_fit(args.fit_json, args.zmode, selected)
    draw_check_image(
        boxes, args.zmode, selected, args.image, args.check_image
    )
    print(f"\nSaved {args.zmode} fit to {args.fit_json}")
    print(f"Saved check image to {args.check_image}")


if __name__ == "__main__":
    main()
