"""
Augmentation and preprocessing transforms for expert training.

Train: resize → augmentation → normalize → tensor
Val/Test: resize → normalize → tensor

All transforms use Albumentations for speed (numpy-based, not PIL).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization (required for pretrained ResNet50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    resize: int = 512,
    aug_prob: float = 0.7,
    min_downscaling: float = 0.8,
) -> A.Compose:
    """
    Training transforms: resize, augmentation, normalize, to tensor.

    Args:
        resize: Target size (square) for uniform resizing.
        aug_prob: Probability of applying each augmentation group.
        min_downscaling: Minimum scale factor for random downscaling (e.g., 0.8 = down to 80%).
    """
    min_downscaling_aug = -(1 - min_downscaling)  # e.g., -0.2
    max_upscaling_aug = 1 / min_downscaling - 1   # e.g., 0.25

    # Geometric + color augmentations (pick one randomly)
    appearance_augs = [
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
        A.CLAHE(p=1),
        A.Blur(p=1),
        A.RandomBrightnessContrast(p=1),
        A.ColorJitter(p=1),
        A.Downscale(p=1),
        A.HueSaturationValue(p=1),
    ]

    # Scaling augmentations (pick one randomly)
    scaling_augs = [
        A.RandomScale(scale_limit=(min_downscaling_aug, -0.01), p=1),
        A.RandomScale(scale_limit=(0.01, max_upscaling_aug), p=1),
    ]

    return A.Compose([
        A.Resize(resize, resize),
        A.OneOf(appearance_augs, p=aug_prob),
        A.OneOf(scaling_augs, p=aug_prob),
        A.ImageCompression(quality_range=(30, 100), p=aug_prob),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(resize: int = 512) -> A.Compose:
    """
    Validation/test transforms: resize, normalize, to tensor.
    No augmentation — deterministic preprocessing only.
    """
    return A.Compose([
        A.Resize(resize, resize),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])