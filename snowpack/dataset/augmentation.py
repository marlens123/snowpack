import numpy as np
import torch
from torchvision.transforms import v2
import cv2
from typing import List
    
class Transpose(torch.nn.Module):
    def forward(self, image, mask):
        assert image.ndim == 3 and image.shape[0] == 3
        print("Image is in the rigth shape!")
        return image.permute(1, 2, 0), mask.permute(1, 2, 0)
        
def get_transformation(mean: List, std: List):
    """
    Returns training and testing transformations with specified mean and std for grayscale images.

    Args:
        mean: Mean for the channels
        std: Standard deviation for the channels

    Returns:
        tuple: (train_transform, test_transform)
    """

    train_transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(3),  # Convert to 3-channel grayscale image
            # Crop a random portion of image and resize it to a given size.
            v2.RandomResizedCrop(size=1024, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=v2.InterpolationMode.NEAREST),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomRotation(degrees=15, interpolation=v2.InterpolationMode.NEAREST),  # type: ignore
            Transpose(),
            #v2.ToTensor(),
            #v2.Normalize(mean=mean, std=std)
        ]
    
    test_transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(3),
            v2.Resize((1024, 1024)),
            Transpose(),
            #v2.ToTensor(),
            #v2.Normalize(mean=mean, std=std)
        ]

    train_transform = v2.Compose(train_transform_list)
    test_transform = v2.Compose(test_transform_list)
    return train_transform, test_transform

def scale(image: np.ndarray, expected_range: int = 255) -> np.ndarray:
    """
    Scales the pixel values of the given image to be between 0 and expected_range.

    Args:
        image (np.ndarray): The input image to normalize.

    Returns:
        np.ndarray: The normalized image with pixel values between 0 and expected_range.
    """
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * expected_range
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

def create_boundary_mask(mask, revert: bool = False, dilate: bool = False, kernel_size: int = 20):
    mask = mask.numpy()
    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    _, boundary_mask = cv2.threshold(magnitude, 1, 255, cv2.THRESH_BINARY)

    # Convert to 0s and 1s
    binary_boundary = (boundary_mask > 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if dilate:
        binary_boundary = cv2.dilate(binary_boundary, kernel, iterations=1)
    if revert:
        binary_boundary = 1 - binary_boundary

    assert np.unique(binary_boundary).tolist() == [0, 1]
    return binary_boundary