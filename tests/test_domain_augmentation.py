from __future__ import annotations

import unittest

from PIL import Image

from scripts.train_classifier import build_transforms


class DomainAugmentationTests(unittest.TestCase):
    def test_domain_augmentation_is_train_only_and_preserves_tensor_shape(self) -> None:
        train_transform, evaluation_transform = build_transforms(
            112, domain_aug=True
        )
        train_names = [type(operation).__name__ for operation in train_transform.transforms]
        evaluation_names = [
            type(operation).__name__ for operation in evaluation_transform.transforms
        ]

        self.assertIn("WarmYellowCreamShift", train_names)
        self.assertIn("RandomBrightRegionOverexposure", train_names)
        self.assertIn("RandomJPEGCompression", train_names)
        self.assertIn("RandomBackgroundContrast", train_names)
        self.assertEqual(evaluation_names, ["Resize", "ToTensor", "Normalize"])

        image = Image.new("RGB", (80, 90), (220, 205, 120))
        tensor = train_transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 112, 112))

    def test_default_training_transform_does_not_enable_domain_augmentation(self) -> None:
        train_transform, _evaluation_transform = build_transforms(
            112, domain_aug=False
        )
        names = [type(operation).__name__ for operation in train_transform.transforms]

        self.assertNotIn("WarmYellowCreamShift", names)
        self.assertNotIn("RandomBrightRegionOverexposure", names)
        self.assertNotIn("RandomJPEGCompression", names)
        self.assertNotIn("RandomBackgroundContrast", names)


if __name__ == "__main__":
    unittest.main()
