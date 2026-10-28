"""Isaac-independent tests for dual-fixture generator geometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gen_tray_v2 import labeler_grid_center, translated_crop_centers


class GeneratorGeometryTests(unittest.TestCase):
    def test_labeler_grid_uses_columns_from_right(self) -> None:
        grid = ((0.0, 0.0), (50.0, 0.0), (0.0, 40.0), (50.0, 40.0))
        self.assertEqual(labeler_grid_center(grid, 1), (50.0, 0.0))
        self.assertEqual(labeler_grid_center(grid, 5), (50.0, 40.0))
        self.assertEqual(labeler_grid_center(grid, 26), (0.0, 0.0))
        self.assertEqual(labeler_grid_center(grid, 30), (0.0, 40.0))

    def test_translated_crop_centers_cover_both_fixtures(self) -> None:
        geometry = {
            "fixtures": {
                prefix: {
                    "centers": {
                        f"{prefix}{position:02d}": (float(position), 100.0)
                        for position in range(1, 31)
                    }
                }
                for prefix in ("G", "S")
            }
        }
        centers = translated_crop_centers(geometry, (7, -11))
        self.assertEqual(len(centers), 60)
        self.assertEqual(centers["G01"], [8.0, 89.0])
        self.assertEqual(centers["S30"], [37.0, 89.0])


if __name__ == "__main__":
    unittest.main()
