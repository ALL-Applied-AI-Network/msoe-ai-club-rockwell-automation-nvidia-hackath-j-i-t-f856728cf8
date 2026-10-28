"""Tests for the Visual ChangeNet dataset exporter."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_changenet import CSV_COLUMNS, EXPECTED_SLOTS, export_dataset


class ChangeNetExportTests(unittest.TestCase):
    def test_csv_columns_and_image_level_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator_out = root / "out"
            crops = generator_out / "crops"
            crops.mkdir(parents=True)
            images = []

            for image_number in range(4):
                image_id = f"img_{image_number:04d}"
                labels = {}
                for slot in EXPECTED_SLOTS:
                    orig_class = "pass" if slot.endswith("01") else "no_vial"
                    labels[slot] = orig_class
                    (crops / f"{image_id}_{slot}.png").write_bytes(
                        f"{image_id}:{slot}:{orig_class}".encode()
                    )
                images.append({"image": image_id, "labels": labels})

            (generator_out / "labels.json").write_text(
                json.dumps({"images": images}), encoding="utf-8"
            )
            destination = root / "dataset"
            summary = export_dataset(generator_out, destination)

            with (destination / "train.csv").open(encoding="utf-8", newline="") as handle:
                train_reader = csv.DictReader(handle)
                train_rows = list(train_reader)
                self.assertEqual(tuple(train_reader.fieldnames or ()), CSV_COLUMNS)
            with (destination / "valid.csv").open(encoding="utf-8", newline="") as handle:
                valid_reader = csv.DictReader(handle)
                valid_rows = list(valid_reader)
                self.assertEqual(tuple(valid_reader.fieldnames or ()), CSV_COLUMNS)
            with (destination / "meta.csv").open(encoding="utf-8", newline="") as handle:
                meta_rows = list(csv.DictReader(handle))

            self.assertEqual(len(train_rows), 180)
            self.assertEqual(len(valid_rows), 60)
            self.assertEqual(
                {row["image_id"] for row in meta_rows if row["split"] == "train"},
                {"img_0000", "img_0001", "img_0002"},
            )
            self.assertEqual(
                {row["image_id"] for row in meta_rows if row["split"] == "valid"},
                {"img_0003"},
            )
            golden_names = {
                path.name for path in (destination / "images" / "golden").glob("*/*.png")
            }
            self.assertEqual(len(golden_names), 240)
            for row in train_rows + valid_rows:
                golden = (
                    destination
                    / "images"
                    / row["golden_path"]
                    / f"{row['object_name']}.png"
                )
                self.assertTrue(golden.is_file(), golden)

            valid_ids = {"img_0003"}
            self.assertTrue(
                all(row["golden_source_image_id"] not in valid_ids for row in meta_rows)
            )
            for row in meta_rows:
                source = crops / (
                    f"{row['golden_source_image_id']}_{row['golden_source_slot']}.png"
                )
                golden = (
                    destination
                    / "images"
                    / row["golden_path"]
                    / f"{row['object_name']}.png"
                )
                self.assertEqual(golden.read_bytes(), source.read_bytes())
                if row["split"] == "train" and row["slot"].endswith("01"):
                    self.assertNotEqual(row["image_id"], row["golden_source_image_id"])

            self.assertEqual(summary.golden_counts[("train", "G")], 90)
            self.assertEqual(summary.golden_counts[("train", "S")], 90)
            self.assertEqual(summary.golden_counts[("valid", "G")], 30)
            self.assertEqual(summary.golden_counts[("valid", "S")], 30)
            self.assertEqual(summary.fallback_counts[("train", "G")], 87)
            self.assertEqual(summary.fallback_counts[("valid", "G")], 29)


if __name__ == "__main__":
    unittest.main()
