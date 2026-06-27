"""Image transforms. Training aug is documented as DISABLED in v1 (see PLAN.md § Image Preprocessing)."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(image_size: int = 224, resize: int = 256) -> transforms.Compose:
    """RandomCrop + flip. NOTE: In v1 we cache features via `inference_transforms`,
    so this augmentation is effectively not exercised. Documented tradeoff."""
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def inference_transforms(image_size: int = 224, resize: int = 256) -> transforms.Compose:
    """Deterministic Resize -> CenterCrop -> Normalize. Used by cache_features, eval, demo."""
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )