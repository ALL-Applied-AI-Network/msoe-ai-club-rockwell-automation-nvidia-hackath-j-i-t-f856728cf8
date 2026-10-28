"""Undistort a real tray image and overlay fitted slot projections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fisheye import PRINCIPAL_POINT, project_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=Path(r"C:\Users\aamp7\Downloads\real_examples\tray_000001.jpg"),
    )
    parser.add_argument(
        "--camera", type=Path, default=Path("calib/camera_fit.json")
    )
    parser.add_argument(
        "--prims", type=Path, default=Path("calib/twin_prims.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\aamp7\Downloads\fit_check"),
    )
    return parser.parse_args()


def ensure_external_output(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ValueError(f"Refusing to write check images inside the repository: {path}")


def slot_point(boxes: dict, slot: str, zmode: str) -> np.ndarray:
    tube = np.asarray(boxes[f"/TrayInspection/{slot}_tube"], dtype=np.float64)
    xy = tube.mean(axis=0)[:2]
    if zmode == "tube_min":
        z_value = tube[0, 2]
    elif zmode == "tube_max":
        z_value = tube[1, 2]
    elif zmode == "cap_center":
        cap = np.asarray(boxes[f"/TrayInspection/{slot}_cap"], dtype=np.float64)
        z_value = cap[:, 2].mean()
    else:
        raise ValueError(f"Unsupported zmode: {zmode}")
    return np.array([xy[0], xy[1], z_value], dtype=np.float64)


def main() -> None:
    args = parse_args()
    ensure_external_output(args.out)
    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    if camera.get("model") != "opencv_fisheye":
        raise ValueError("camera_fit.json must use model='opencv_fisheye'")
    boxes = json.loads(args.prims.read_text(encoding="utf-8"))
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    width, height = (int(value) for value in camera["image_size"])
    if (image.shape[1], image.shape[0]) != (width, height):
        raise ValueError(
            f"Expected image size {(width, height)}, got {(image.shape[1], image.shape[0])}"
        )
    focal = float(camera["focal_px"])
    k1 = float(camera["k1"])
    k2 = float(camera["k2"])
    cx, cy = PRINCIPAL_POINT
    intrinsic = np.array(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    new_intrinsic = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsic,
        distortion,
        (width, height),
        np.eye(3),
        balance=0.72,
        new_size=(width, height),
        fov_scale=1.0,
    )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        intrinsic,
        distortion,
        np.eye(3),
        new_intrinsic,
        (width, height),
        cv2.CV_32FC1,
    )
    undistorted = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    slot_names = [f"G{index:02d}" for index in range(1, 31)]
    world = np.vstack(
        [slot_point(boxes, slot, camera["zmode"]) for slot in slot_names]
    )
    pixels = project_world(
        world,
        np.asarray(camera["R_world_to_cv"], dtype=np.float64),
        np.asarray(camera["position_m"], dtype=np.float64),
        focal,
        k1,
        k2,
    )
    overlay = image.copy()
    for index, pixel in enumerate(pixels, start=1):
        center = tuple(np.rint(pixel).astype(int))
        cv2.circle(overlay, center, 25, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            (center[0] - 13, center[1] + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    undistorted_path = args.out / f"{stem}_undistorted.png"
    overlay_path = args.out / f"{stem}_fit2.png"
    if not cv2.imwrite(str(undistorted_path), undistorted):
        raise OSError(f"Could not write image: {undistorted_path}")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not write image: {overlay_path}")
    print(f"wrote {undistorted_path}")
    print(f"wrote {overlay_path}")


if __name__ == "__main__":
    main()
