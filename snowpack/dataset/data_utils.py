import rasterio
import os
import cv2
import numpy as np

def load_tifs_resize_to_np(image_path: str, mask_path: str, size: int = (1024, 1024)):

    images = []
    masks = []

    for _, i in enumerate(os.listdir(image_path)):
        if '.DS_Store' in i:
            continue
        image = rasterio.open(image_path + i).read(1)
        images.append(cv2.resize(image, size))

    for _, i in enumerate(os.listdir(mask_path)):
        if '.DS_Store' in i:
            continue
        mask= rasterio.open(mask_path + i).read(1)
        # use nearest neighbour interpolation to make sure that masks remain categorical
        masks.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))

    image_np = np.array(images)
    mask_np = np.array(masks)

    return image_np, mask_np


def load_tifs_resize_to_np_retain_ratio(image_path: str, mask_path: str, size: int = (1024, 1024)):

    images = []
    masks = []

    for _, i in enumerate(os.listdir(image_path)):
        if '.DS_Store' in i:
            continue
        image = rasterio.open(image_path + i).read(1)
        r = np.min([size[0] / image.shape[1], size[0] / image.shape[0]])
        images.append(cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r))))

    for _, i in enumerate(os.listdir(mask_path)):
        if '.DS_Store' in i:
            continue
        mask= rasterio.open(mask_path + i).read(1)
        r = np.min([size[0] / mask.shape[1], size[0] / mask.shape[0]])
        # use nearest neighbour interpolation to make sure that masks remain categorical
        masks.append(cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST))

    return images, masks

import numpy as np
import torch

def recombine_patches(probability_class_masks, patch_coords, image_size):
    """
    Recombines overlapping probability class masks into a single probability map for the entire image.

    Args:
        probability_class_masks (list of torch.Tensor): List of predicted probability maps for each patch.
            Each tensor has shape (C, patch_height, patch_width), where C is the number of classes.
        patch_coords (list of tuple): List of top-left (x, y) coordinates for each patch.
        image_size (tuple): (height, width) of the original image.

    Returns:
        torch.Tensor: Recombined probability map of shape (C, image_height, image_width).
    """
    num_classes = probability_class_masks[0].shape[0]
    height, width = image_size

    # Initialize probability map and overlap count map
    combined_probabilities = torch.zeros((num_classes, height, width), dtype=probability_class_masks[0].dtype)
    overlap_count = torch.zeros((height, width), dtype=torch.float32)

    for patch, (x, y) in zip(probability_class_masks, patch_coords):
        patch_height, patch_width = patch.shape[1], patch.shape[2]

        # Add probabilities to the appropriate region in the combined map
        combined_probabilities[:, y:y + patch_height, x:x + patch_width] += patch

        # Track overlap count
        overlap_count[y:y + patch_height, x:x + patch_width] += 1

    # Avoid division by zero
    overlap_count = overlap_count.clamp(min=1)

    # Normalize probabilities by the overlap count
    combined_probabilities /= overlap_count.unsqueeze(0)  # Add channel dimension to match

    return combined_probabilities

def generate_patch_coords(image_size, patch_size, overlap):
    """
    Generates top-left coordinates for patches based on image size, patch size, and overlap.

    Args:
        image_size (tuple): (height, width) of the full image.
        patch_size (int): Size of each patch (assumes square patches).
        overlap (int): Desired overlap in pixels.

    Returns:
        list of tuple: List of (x, y) coordinates for the top-left corner of each patch.
    """
    stride = patch_size - overlap
    height, width = image_size

    patch_coords = [
        (x, y)
        for y in range(0, height - patch_size + stride, stride)
        for x in range(0, width - patch_size + stride, stride)
    ]

    return patch_coords

def generate_final_mask(combined_probabilities):
    """
    Generates the final mask by taking the argmax of the recombined probability map.

    Args:
        combined_probabilities (torch.Tensor): Recombined probability map of shape (C, height, width).

    Returns:
        torch.Tensor: Final mask of shape (height, width).
    """
    final_mask = torch.argmax(combined_probabilities, dim=0)
    return final_mask