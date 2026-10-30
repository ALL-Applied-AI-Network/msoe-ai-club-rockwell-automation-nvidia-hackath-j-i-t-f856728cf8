from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.define_slots import DEFAULT_CROP_OFFSETS, slot_document
from scripts.model_preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from scripts.predict import (
    DEFAULT_THRESHOLD,
    SlotROI,
    extract_crops,
    load_slots,
    process_frame,
    predictions_from_probabilities,
    probabilities_from_logits,
)


class FactoryPredictionTests(unittest.TestCase):
    def test_factory_default_threshold_is_validation_start_value(self) -> None:
        self.assertEqual(DEFAULT_THRESHOLD, 0.70)

    def test_checked_in_slots_cover_all_g_positions_with_training_crop_size(self) -> None:
        slots_path = Path(__file__).resolve().parents[1] / "slots_G.json"

        slots = load_slots(slots_path)

        self.assertEqual([slot.slot for slot in slots], [f"G{i:02d}" for i in range(1, 31)])
        self.assertTrue(all((slot.w, slot.h) == (80, 90) for slot in slots))

    def test_extracts_a_few_sample_crops_and_converts_bgr_to_rgb(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        image[:, 0:10] = (10, 20, 30)
        image[:, 10:20] = (40, 50, 60)
        image[:, 20:30] = (70, 80, 90)
        slots = [
            SlotROI("G01", 0, 0, 10, 20),
            SlotROI("G02", 10, 0, 10, 20),
            SlotROI("G03", 20, 0, 10, 20),
        ]

        crops = extract_crops(image, slots)

        self.assertEqual([crop.size for crop in crops], [(10, 20)] * 3)
        self.assertEqual(crops[0].getpixel((0, 0)), (30, 20, 10))
        self.assertEqual(crops[1].getpixel((0, 0)), (60, 50, 40))
        self.assertEqual(crops[2].getpixel((0, 0)), (90, 80, 70))

    def test_defect_flag_uses_pass_probability_not_top_class_only(self) -> None:
        slots = [
            SlotROI("G01", 0, 0, 1, 1),
            SlotROI("G02", 1, 0, 1, 1),
            SlotROI("G03", 2, 0, 1, 1),
        ]
        probabilities = np.asarray(
            [
                [0.89, 0.04, 0.03, 0.02, 0.02],
                [0.91, 0.03, 0.02, 0.02, 0.02],
                [0.20, 0.70, 0.04, 0.03, 0.03],
            ]
        )

        predictions = predictions_from_probabilities(slots, probabilities, 0.90)

        self.assertEqual(predictions[0].predicted_class, "pass")
        self.assertEqual(predictions[0].flag, "DEFECT")
        self.assertEqual(predictions[1].flag, "PASS")
        self.assertEqual(predictions[2].predicted_class, "missing_cap")
        self.assertEqual(predictions[2].flag, "DEFECT")

    def test_logits_are_converted_to_five_class_probabilities(self) -> None:
        probabilities = probabilities_from_logits(
            np.asarray([[3.0, 1.0, 0.0, -1.0, -2.0]] * 3)
        )

        self.assertEqual(probabilities.shape, (3, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
        self.assertTrue(np.all(np.argmax(probabilities, axis=1) == 0))

    def test_define_slots_writes_training_crop_offsets(self) -> None:
        centers = [(100 + index, 100 + index) for index in range(30)]

        document = slot_document(500, 500, centers, DEFAULT_CROP_OFFSETS)

        self.assertEqual(document["crop_size"], [80, 90])
        self.assertEqual(document["slots"][0], {"slot": "G01", "x": 62, "y": 53, "w": 80, "h": 90})
        self.assertEqual(document["slots"][-1]["slot"], "G30")

    def test_preprocessing_constants_match_training_configuration(self) -> None:
        self.assertEqual(IMAGE_SIZE, 112)
        self.assertEqual(IMAGENET_MEAN, (0.485, 0.456, 0.406))
        self.assertEqual(IMAGENET_STD, (0.229, 0.224, 0.225))

    def test_process_frame_writes_thirty_row_outputs_with_fake_classifier(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        out_dir = Path(temporary_directory.name)
        image = np.zeros((5, 6, 3), dtype=np.uint8)
        slots = [
            SlotROI(f"G{index + 1:02d}", index % 6, index // 6, 1, 1)
            for index in range(30)
        ]

        def fake_classifier(_crops):
            probabilities = np.zeros((30, 5), dtype=np.float64)
            probabilities[:, 0] = 0.95
            probabilities[:, 1] = 0.05
            probabilities[-1] = [0.10, 0.80, 0.04, 0.03, 0.03]
            return probabilities

        predictions = process_frame(
            "sample.png", image, slots, fake_classifier, 0.90, out_dir
        )

        with (out_dir / "sample_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            csv_rows = list(csv.DictReader(handle))
        json_document = json.loads(
            (out_dir / "sample_predictions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(predictions), 30)
        self.assertEqual(len(csv_rows), 30)
        self.assertEqual(len(json_document["predictions"]), 30)
        self.assertEqual(csv_rows[-1]["flag"], "DEFECT")
        self.assertTrue((out_dir / "sample_overlay.png").is_file())


if __name__ == "__main__":
    unittest.main()
