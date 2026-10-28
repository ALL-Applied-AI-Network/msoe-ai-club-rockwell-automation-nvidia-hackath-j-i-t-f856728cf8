"""Pure OpenCV-fisheye projection and inversion helpers."""

from __future__ import annotations

import math

import numpy as np


PRINCIPAL_POINT = (960.0, 540.0)
MAX_PINHOLE_THETA = math.radians(88.0)


def distort_theta(theta, k1: float, k2: float):
    theta = np.asarray(theta, dtype=np.float64)
    theta2 = theta * theta
    return theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2)


def distortion_derivative(theta, k1: float, k2: float):
    theta = np.asarray(theta, dtype=np.float64)
    theta2 = theta * theta
    return 1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta2 * theta2


def invert_theta(
    theta_d,
    k1: float,
    k2: float,
    *,
    iterations: int = 30,
    upper: float = MAX_PINHOLE_THETA,
):
    """Invert a monotonic theta polynomial with safeguarded Newton steps."""
    target = np.asarray(theta_d, dtype=np.float64)
    maximum = float(distort_theta(upper, k1, k2))
    if np.any(target < -1.0e-12) or np.any(target > maximum + 1.0e-9):
        raise ValueError("Distorted angle lies outside the monotonic fisheye domain")

    low = np.zeros_like(target)
    high = np.full_like(target, upper)
    theta = np.clip(target, 0.0, upper)
    for _ in range(iterations):
        mapped = distort_theta(theta, k1, k2)
        low = np.where(mapped < target, theta, low)
        high = np.where(mapped >= target, theta, high)
        derivative = distortion_derivative(theta, k1, k2)
        safe_derivative = np.maximum(derivative, 1.0e-9)
        newton = theta - (mapped - target) / safe_derivative
        midpoint = (low + high) / 2.0
        converged = np.abs(mapped - target) < 1.0e-14
        use_newton = (
            (newton >= low)
            & (newton <= high)
            & np.isfinite(newton)
            & (derivative > 0.0)
        )
        theta = np.where(converged, theta, np.where(use_newton, newton, midpoint))
    return theta


def corner_theta_max(
    image_size: tuple[int, int] | list[int],
    focal_px: float,
    k1: float,
    k2: float,
    *,
    margin: float = 1.10,
    principal_point: tuple[float, float] = PRINCIPAL_POINT,
) -> float:
    width, height = image_size
    cx, cy = principal_point
    corner_radii = [
        math.hypot(u - cx, v - cy)
        for u, v in ((0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0))
    ]
    largest_theta_d = max(corner_radii) / focal_px
    corner_theta = float(invert_theta(largest_theta_d, k1, k2))
    theta_max = corner_theta * margin
    if theta_max >= MAX_PINHOLE_THETA:
        raise ValueError("Required fisheye field of view is too wide for a pinhole render")
    return theta_max


def minimum_derivative(k1: float, k2: float, theta_max: float) -> float:
    samples = np.linspace(0.0, theta_max, 1025)
    return float(np.min(distortion_derivative(samples, k1, k2)))


def fisheye_pixels_to_pinhole_normalized(
    pixels: np.ndarray,
    focal_px: float,
    k1: float,
    k2: float,
    *,
    principal_point: tuple[float, float] = PRINCIPAL_POINT,
) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    cx, cy = principal_point
    delta = pixels - np.array([cx, cy], dtype=np.float64)
    radius_px = np.linalg.norm(delta, axis=-1)
    theta_d = radius_px / focal_px
    theta = invert_theta(theta_d, k1, k2)
    pinhole_radius = np.tan(theta)
    scale = np.zeros_like(radius_px)
    nonzero = radius_px > 1.0e-12
    scale[nonzero] = pinhole_radius[nonzero] / radius_px[nonzero]
    normalized = delta * scale[..., None]
    normalized[~nonzero] = 0.0
    return normalized


def pinhole_normalized_to_fisheye_pixels(
    normalized: np.ndarray,
    focal_px: float,
    k1: float,
    k2: float,
    *,
    principal_point: tuple[float, float] = PRINCIPAL_POINT,
) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float64)
    radius = np.linalg.norm(normalized, axis=-1)
    theta = np.arctan(radius)
    theta_d = distort_theta(theta, k1, k2)
    scale = np.ones_like(radius)
    nonzero = radius > 1.0e-12
    scale[nonzero] = theta_d[nonzero] / radius[nonzero]
    distorted = normalized * scale[..., None]
    return focal_px * distorted + np.asarray(principal_point, dtype=np.float64)


def project_world(
    points: np.ndarray,
    rotation_world_to_cv: np.ndarray,
    position_m: np.ndarray,
    focal_px: float,
    k1: float,
    k2: float,
    *,
    principal_point: tuple[float, float] = PRINCIPAL_POINT,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    camera_points = (
        np.asarray(rotation_world_to_cv, dtype=np.float64)
        @ (points - np.asarray(position_m, dtype=np.float64)).T
    ).T
    if np.any(camera_points[:, 2] <= 1.0e-9):
        raise ValueError("One or more points project behind the camera")
    normalized = camera_points[:, :2] / camera_points[:, 2, None]
    return pinhole_normalized_to_fisheye_pixels(
        normalized, focal_px, k1, k2, principal_point=principal_point
    )
