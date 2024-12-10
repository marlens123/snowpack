from torch.utils.data import Dataset as BaseDataset
from patchify import patchify
import numpy as np
import torch
import cv2
from snowpack.dataset.augmentation import create_boundary_mask
from torchvision.tv_tensors import Mask

class SnowDataset(BaseDataset):
    def __init__(
        self,
        images,
        masks, 
        transform=None, 
        transform_images=None,
        transform_masks_and_images=None,
        size_strategy: str = 'resize_simple',
        boundary_mask: bool = True,
        revert: bool = False,
        kernel_size: int = 20,
        num_points: int = 50,
        resized_image_size: tuple = (1024, 1024),
        dilate: bool = False,
    ):
        assert len(images) == len(masks)
        assert size_strategy in ['resize_simple', 'resize_retain_aspect', 'row_chunks', 'single_chunk']

        self.transform = transform
        if transform_images:
            self.transform_images = transform_images
            self.transform_masks_and_images = transform_masks_and_images
        else:
            self.transform_images = None
            self.transform_masks_and_images = None
        self.num_points = num_points

        self.size_strategy = size_strategy
        self.resized_image_size = resized_image_size
        self.boundary_mask = boundary_mask
        self.dilate = dilate
        self.revert = revert
        self.kernel_size = kernel_size

        self.images = images
        self.masks = masks

        print("Number of images: " + str(len(self.images)))
        print("Number of masks: " + str(len(self.masks)))

    @staticmethod
    def get_points(binarized_mask, num_points=50):
        points = []

        # Get all coordinates inside the eroded mask and choose a random point
        coords = np.argwhere(binarized_mask > 0)
        if len(coords) > 0:
            for _ in range(num_points):  # Select as many points as there are unique labels
                yx = np.array(coords[np.random.randint(len(coords))])
                points.append([yx[1], yx[0]])

        points = np.array(points)

        return points


    def expand_grayscale_channel(self, image):
        # From (H, W) to (H, W, 3) to match the shape of the data the model was pre-trained on
        image = np.expand_dims(image, -1)
        image = image.repeat(3, axis=-1)

        assert image[0].all() == image[1].all() == image[2].all()

        return image
    
    
    @staticmethod
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        #image = self.normalize_image(image)

        if self.transform is not None or self.transform_images is not None:
            mask = np.expand_dims(mask, axis=0).astype(np.float32)
            if self.transform_images is not None:
                mask = Mask(mask)
                image, mask = self.transform_masks_and_images(image, mask)
                image = self.transform_images(image)
            else:
                image, mask = self.transform(image, mask)
            mask = mask.permute(2, 0, 1)
        else:
            image = self.expand_grayscale_channel(image).astype(np.float32)
            mask = np.array(mask)
            mask = np.expand_dims(mask, axis=0).astype(np.float32)

        if self.boundary_mask:
            mask = self.create_boundary_mask(mask, revert=self.revert, dilate=self.dilate, kernel_size=self.kernel_size)
            #prompt = np.expand_dims(self.get_points(mask, num_points=self.num_points), axis=1)
    
        # from: https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1_training/sam2.1_hiera_b%2B_MOSE_finetune.yaml
        # mean: [0.485, 0.456, 0.406]
        # std: [0.229, 0.224, 0.225]

        return np.array(image).astype(np.float32), np.array(mask).astype(np.float32)
