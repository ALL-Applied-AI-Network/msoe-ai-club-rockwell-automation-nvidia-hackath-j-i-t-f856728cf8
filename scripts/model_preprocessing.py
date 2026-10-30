"""Shared classifier preprocessing used by training and factory inference."""

from __future__ import annotations


IMAGE_SIZE = 112
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_evaluation_transform(image_size: int = IMAGE_SIZE):
    """Build the deterministic transform used for validation, test, and inference."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
