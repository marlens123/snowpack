from torch.utils.data import Dataset as BaseDataset
import albumentations as A
from patchify import patchify
from pathlib import Path
import numpy as np
import torch
import cv2

class SnowDataset(BaseDataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray, 
        split_mode: str,
        size_strategy: str = 'resize',
        target_image_size: tuple = (1024, 1024),
    ):
        assert len(images) == len(masks)
        assert split_mode in ['train', 'test']
        assert size_strategy in ['resize', 'row_chunks', 'single_chunk']

        self.split_mode = split_mode
        self.size_strategy = size_strategy
        self.target_image_size = target_image_size

        self.input_images = images
        self.input_masks = masks
        self.target_images = []
        self.target_masks = []

        # preprocess images and masks and store them in target_images and target_masks
        for i in range(len(images)):
            self.preprocess(images[i], masks[i], self.size_strategy)
    
    def preprocess(self, image, mask):

        resized_images = []
        resized_masks = []

        if "chunks" in self.size_strategy:
             # idea: during inference, patchify then unpatchify the image
             # for training, patchify the image and mask row wise to avoid too homogeneous patches
            resized_images, resized_masks = self.chunk_row_wise(image, mask)

        elif self.size_strategy == "resize":
            # problem with resizing: 2-times interpolation, which can lead to loss of information
            resized_images.append(cv2.resize(image, self.target_image_size))
            resized_masks.append(cv2.resize(mask, self.target_image_size, interpolation=cv2.INTER_NEAREST))

        # apply rest of preprocessing pipeline chunk-wise
        for i in range(len(resized_images)):
            image = resized_images[i]
            mask = resized_masks[i]

            # augment after potential chunking to have more diverse patches
            if self.split_mode == "train":
                augment = self.get_augmentations()
                image, mask = augment(image=image, mask=mask)

            # TODO: Implement normalization
            # normalize values after augmentation to retain integrity of the normalization
            image, mask = self.normalize(image, mask)

            image = self.expand_grayscale_channel(image)
            # permute dimensions to match torch format (C, H, W), convert to float for precision
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            # adds channel dimension to mask
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            self.target_images.append(image)
            self.target_masks.append(mask)

    
    def expand_grayscale_channel(self, image):
        # From (H, W) to (H, W, 3) to match the shape of the data the model was pre-trained on
        image = np.expand_dims(image, -1)
        image = image.repeat(3, axis=-1)
        return image

    # TODO: Implement normalization
    def normalize(self):
        # Normalize the images based on the statistics of the pre-training dataset
        pass

    def get_augmentations(self, sample):
        # Augmentation techniques: horizontal flip, vertical flip, rotate by 10-20 degrees
        train_transform = A.RandomOrder([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # interpolation=0 means nearest neighbour interpolation, necessary to keep the mask values as integers
            A.Rotate(limit=10, interpolation=0, p=0.5),
        ], p=1.0)

        # composes multiple transformations together to apply them sequentially in the order they are passed,
        # probability of applying all transformations is 1 per default
        return A.Compose(train_transform)

    def chunk_row_wise(self, image, mask, target_image_size):
        # first extract single column from image and mask, then chunk it
        # shape of image and mask: (H, W)
        # shape of resulting patches: (n_patches, 1, target_image_size, target_image_size)

        # random column idx
        # note: this will never consider the rightmost pixel columns of the image, if the image is not divisible by the target_image_size
        max_idx = int(image.shape[1] / target_image_size)
        chunk_column_idx = np.random.randint(0, max_idx)

        image_column = image[:, chunk_column_idx]
        mask_column = mask[:, chunk_column_idx]
        # step defines the overlap / offset between patches
        image_column_patches = patchify(image_column, (target_image_size, target_image_size), step=1)
        mask_column_patches = patchify(mask_column, (target_image_size, target_image_size), step=1)

        print(image_column_patches.shape)
        print(mask_column_patches.shape)

        for j in range(image_column_patches.shape[0]):
            image_patch = image_column_patches[j, 0]
            mask_patch = mask_column_patches[j, 0]

            # append image and mask patch to list
            self.target_images.append(image_patch)
            self.target_masks.append(mask_patch)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.target_images[idx]
        mask = self.target_masks[idx]

        assert image.dtype == torch.float32
        assert mask.dtype == torch.float32

        return (image, mask)