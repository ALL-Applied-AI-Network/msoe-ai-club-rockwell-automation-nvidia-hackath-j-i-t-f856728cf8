from __future__ import annotations

import unittest

import numpy as np

from scripts.eval_real import evaluate_probabilities


class RealEvaluationTests(unittest.TestCase):
    def test_fake_probabilities_report_confusion_and_binary_errors(self) -> None:
        targets = [0, 0, 1, 2, 3, 4]
        probabilities = np.asarray(
            [
                [0.90, 0.03, 0.03, 0.02, 0.02],
                [0.40, 0.35, 0.10, 0.10, 0.05],
                [0.20, 0.70, 0.04, 0.03, 0.03],
                [0.75, 0.05, 0.10, 0.05, 0.05],
                [0.10, 0.05, 0.05, 0.75, 0.05],
                [0.05, 0.05, 0.05, 0.05, 0.80],
            ]
        )

        report = evaluate_probabilities(targets, probabilities, threshold=0.70)

        self.assertEqual(report["sample_count"], 6)
        self.assertEqual(report["confusion_matrix"][0][0], 2)
        self.assertEqual(report["confusion_matrix"][2][0], 1)
        self.assertEqual(report["per_class_recall"]["missing_cap"], 1.0)
        self.assertEqual(report["per_class_recall"]["crooked_cap"], 0.0)
        self.assertEqual(report["binary"]["defects_missed"], 1)
        self.assertEqual(report["binary"]["false_alarms_on_pass"], 1)
        self.assertEqual(len(report["threshold_sweep"]), 9)
        self.assertEqual(report["threshold_sweep"][0]["threshold"], 0.1)
        self.assertEqual(report["threshold_sweep"][-1]["threshold"], 0.9)


if __name__ == "__main__":
    unittest.main()
