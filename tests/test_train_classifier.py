from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.classifier_utils import load_manifest, select_tau, write_probability_csv


class ClassifierUtilityTests(unittest.TestCase):
    def test_load_manifest_filters_split_and_resolves_paths(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        image_path = root / "data" / "train" / "pass" / "crop.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"test")
        validation_path = root / "data" / "val" / "pass" / "crop.png"
        validation_path.parent.mkdir(parents=True)
        validation_path.write_bytes(b"test")
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "path",
                    "label",
                    "split",
                    "job",
                    "image_id",
                    "slot",
                    "damage_type",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "path": "data/train/pass/crop.png",
                    "label": "pass",
                    "split": "train",
                    "job": "job-a",
                    "image_id": "image-1",
                    "slot": "G01",
                    "damage_type": "",
                }
            )
            writer.writerow(
                {
                    "path": "data/val/pass/crop.png",
                    "label": "pass",
                    "split": "val",
                    "job": "job-a",
                    "image_id": "image-2",
                    "slot": "G02",
                    "damage_type": "",
                }
            )

        rows = load_manifest(root, "train")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], "image-1")
        self.assertEqual(Path(rows[0]["absolute_path"]), image_path)

    def test_load_manifest_reports_all_missing_files_across_splits(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        fieldnames = [
            "path",
            "label",
            "split",
            "job",
            "image_id",
            "slot",
            "damage_type",
        ]
        with (root / "manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for split, filename in (
                ("train", "missing_train.png"),
                ("val", "missing_val.png"),
                ("test", "missing_test.png"),
            ):
                writer.writerow(
                    {
                        "path": f"data/{split}/pass/{filename}",
                        "label": "pass",
                        "split": split,
                        "job": "job-a",
                        "image_id": split,
                        "slot": "G01",
                        "damage_type": "",
                    }
                )

        with self.assertRaisesRegex(
            FileNotFoundError, r"Manifest references 3 missing file\(s\)"
        ) as context:
            load_manifest(root, "train")

        message = str(context.exception)
        self.assertIn("missing_train.png", message)
        self.assertIn("missing_val.png", message)
        self.assertIn("missing_test.png", message)

    def test_load_manifest_accepts_manifest_outside_data_root(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        data_root = root / "extracted"
        image_path = data_root / "data" / "train" / "pass" / "crop.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"test")
        manifest_path = root / "input" / "manifest.csv"
        manifest_path.parent.mkdir()
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "path",
                    "label",
                    "split",
                    "job",
                    "image_id",
                    "slot",
                    "damage_type",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "path": "data/train/pass/crop.png",
                    "label": "pass",
                    "split": "train",
                    "job": "job-a",
                    "image_id": "image-1",
                    "slot": "G01",
                    "damage_type": "",
                }
            )

        rows = load_manifest(data_root, "train", manifest_path)

        self.assertEqual(Path(rows[0]["absolute_path"]), image_path)

    def test_select_tau_meets_recall_with_lowest_false_alarm_rate(self) -> None:
        probabilities = [
                [0.90, 0.04, 0.03, 0.02, 0.01],
                [0.55, 0.20, 0.15, 0.09, 0.01],
                [0.70, 0.10, 0.10, 0.09, 0.01],
                [0.20, 0.60, 0.10, 0.09, 0.01],
        ]
        targets = [0, 0, 1, 2]

        tau, numbers = select_tau(probabilities, targets, minimum_recall=1.0)

        self.assertGreater(tau, 0.70)
        self.assertLess(tau, 0.90)
        self.assertEqual(numbers["recall"], 1.0)
        self.assertEqual(numbers["false_alarm_rate"], 0.5)

    def test_select_tau_ignores_no_vial_samples(self) -> None:
        probabilities = [
                [0.80, 0.05, 0.05, 0.05, 0.05],
                [0.95, 0.01, 0.01, 0.01, 0.02],
                [0.40, 0.50, 0.04, 0.03, 0.03],
        ]
        targets = [0, 4, 1]

        _tau, numbers = select_tau(probabilities, targets, minimum_recall=1.0)

        self.assertEqual(numbers["true_positive"], 1)
        self.assertEqual(numbers["false_positive"], 0)

    def test_write_probability_csv_records_every_validation_probability(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        output_path = Path(temporary_directory.name) / "validation_predictions.csv"
        rows = [
            {
                "path": "data/val/pass/a.png",
                "label": "pass",
                "job": "job-a",
                "image_id": "a",
                "slot": "G01",
                "damage_type": "",
            },
            {
                "path": "data/val/missing_cap/b.png",
                "label": "missing_cap",
                "job": "job-b",
                "image_id": "b",
                "slot": "G02",
                "damage_type": "",
            },
        ]
        probabilities = [
            [0.05, 0.85, 0.04, 0.03, 0.03],
            [0.80, 0.10, 0.04, 0.03, 0.03],
        ]

        write_probability_csv(
            output_path,
            rows,
            probabilities,
            targets=[1, 0],
            indices=[1, 0],
        )

        with output_path.open(newline="", encoding="utf-8") as handle:
            output_rows = list(csv.DictReader(handle))
        self.assertEqual(len(output_rows), 2)
        self.assertEqual(output_rows[0]["path"], rows[1]["path"])
        self.assertEqual(output_rows[0]["target_class"], "missing_cap")
        self.assertEqual(output_rows[0]["predicted_class"], "missing_cap")
        self.assertEqual(output_rows[0]["p_missing_cap"], "0.8500000000")
        self.assertEqual(output_rows[1]["p_pass"], "0.8000000000")


if __name__ == "__main__":
    unittest.main()
