"""Numerical tests for the monotonic OpenCV-fisheye mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fisheye import (
    fisheye_pixels_to_pinhole_normalized,
    pinhole_normalized_to_fisheye_pixels,
)


class FisheyeRoundTripTests(unittest.TestCase):
    def test_forward_inverse_grid_is_subpixel_accurate(self) -> None:
        x_values = np.linspace(0.0, 1919.0, 17)
        y_values = np.linspace(0.0, 1079.0, 11)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        focal = 1050.0
        k1 = 0.08
        k2 = -0.015

        normalized = fisheye_pixels_to_pinhole_normalized(
            pixels, focal, k1, k2
        )
        reconstructed = pinhole_normalized_to_fisheye_pixels(
            normalized, focal, k1, k2
        )
        errors = np.linalg.norm(reconstructed - pixels, axis=1)
        self.assertLess(float(errors.max()), 0.01)


if __name__ == "__main__":
    unittest.main()
