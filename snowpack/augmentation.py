import numpy as np
import torch
from torchvision.transforms import v2


def get_transformation(mean, std):
    """
    Returns training and testing transformations with specified mean and std for grayscale images.

    Args:
        mean: Mean for the channels
        std: Standard deviation for the channels

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = v2.Compose(
        [
            normalize_image,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(3),  # Convert to 3-channel grayscale image
            # Crop a random portion of image and resize it to a given size.
            v2.RandomResizedCrop(size=1024, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomRotation(degrees=15),  # type: ignore
            v2.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = v2.Compose(
        [
            normalize_image,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(3),
            v2.Resize((1024, 1024)),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, test_transform


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the pixel values of the given image to be between 0 and 1.

    Args:
        image (np.ndarray): The input image to normalize.

    Returns:
        np.ndarray: The normalized image with pixel values between 0 and 1.
    """
    min_val = image.min(axis=(0, 1), keepdims=True)
    max_val = image.max(axis=(0, 1), keepdims=True)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def revert_normalization(image: torch.Tensor, mean, std):
    """
    This function reverts the normalization applied to the image.
    Normalization is needed in the `transformation` pipeline and it ensure that the pixel values are between -1 and 1.
    But for visualization, we need to revert the normalization to get the original pixel values, which are between 0 and 1.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    reverted_image = image * std + mean
    return reverted_image
