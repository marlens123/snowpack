import math
import os
from typing import List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask

from snowpack.dataset.augmentation import scale, create_boundary_mask


class DynamicImagePatchesDataset(Dataset):
    def __init__(
            self, 
            data_dir: str, 
            patch_size: int, 
            overlap: int = 0, 
            inference_mode: bool = False, 
            transform=None, 
            num_points: int = 50, 
            boundary_mask: bool = False,
            revert: bool = False,
            dilate: bool = False,
            kernel_size: int = 20
        ):
        """
        Args:
            image_dir (str): Path to the directory containing data (images and masks).
            patch_size (int): Size of each patch.
            overlap (int): Number of pixels to overlap between patches.
            inference_mode (bool): If True, the dataset will be created in test mode, which means no masks are needed.
            transform (callable, optional): Optional transform to apply to patches.
        NOTICE:
        This dataset assumes that `data_dir` is pointing to a directory that has two sub-directories called: masks and images.
        The images and masks should have the same name but the masks should end with `_mask`, and both images and masks are to be of
        `.tiff` format.
        """
        # load both masks and images
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.inference_mode = inference_mode

        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.num_points = num_points
        self.boundary_mask = boundary_mask
        self.revert = revert
        self.dilate = dilate
        self.kernel_size = kernel_size

        # list all the images and masks in the dataset and normalize them
        self.image_paths = [
            os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith(".tiff")
        ]
        # build the mask paths based on the name of the file existing in the self.image_paths so that we make sure
        # there is an allignmnet between the images and the masks i.e. the mask of the first image is the first mask
        if not inference_mode:
            self.mask_paths = [
                image_path.replace(".tiff", "_mask.tiff").replace("/images/", "/masks/")
                for image_path in self.image_paths
            ]
        self.image_paths = self.preprocess_and_save_image(self.image_paths)
        if not inference_mode:
            self.mask_paths = self.preprocess_and_save_image(self.mask_paths)

        if not inference_mode:
            assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must be the same."
            self._confirm_image_and_mask_alignment()

        # compute the number of patches each image and mask will generate

        # we can do this only once as the number of patchs for images and masks are the same
        self.image_patch_counts = self._compute_patch_counts()
        self.index_map = self._create_index_map()

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

    def _confirm_image_and_mask_alignment(self):
        """Confirm the size, i.e. wdith and height, of each image and mask are the same."""
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            assert (
                image.size == mask.size
            ), f"Image and mask sizes must match, image: {image.size}, mask: {mask.size} for {image_path} and {mask_path}"
        print("All images and masks are aligned.")

    def preprocess_and_save_image(self, image_paths) -> List[str]:
        preprocessed_image_paths = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = np.array(image)
            image = scale(image)
            image = Image.fromarray(image)
            image = image.convert("L")
            new_image_path = image_path.replace(".tiff", ".jpg")
            preprocessed_image_paths.append(new_image_path)
            image.save(new_image_path)
        return preprocessed_image_paths
    
    def preprocess_and_save_mask(self, mask_paths) -> List[str]:
        preprocessed_mask_paths = []
        for mask_path in mask_paths:
            mask = Image.open(mask_path)
            mask = np.array(mask)
            if self.boundary_masks:
                mask = create_boundary_mask(mask, revert=self.revert, dilate=self.dilate, kernel_size=self.kernel_size)
            mask = Image.fromarray(mask)
            mask = mask.convert("L")
            new_mask_path = mask_path.replace(".tiff", ".jpg")
            preprocessed_mask_paths.append(new_mask_path)
            mask.save(new_mask_path)
        return preprocessed_mask_paths

    def _compute_patch_counts(self):
        """Precompute the number of patches each image will generate."""
        patch_counts = []
        for img_path in self.image_paths:
            image = Image.open(img_path)
            width, height = image.size

            # Adjust steps for overlap
            step_size = self.patch_size - self.overlap
            count_x = math.ceil((width - self.overlap) / step_size)
            count_y = math.ceil((height - self.overlap) / step_size)

            patch_counts.append(count_x * count_y)
        return patch_counts

    def _create_index_map(self):
        """Create a global index map for quick lookup."""
        index_map = {}
        global_index = 0
        for image_index, patch_count in enumerate(self.image_patch_counts):
            for patch_index in range(patch_count):
                index_map[global_index] = (image_index, patch_index)
                global_index += 1
        return index_map

    def __len__(self):
        """Total number of patches across all images."""
        return sum(self.image_patch_counts)

    def get_patch(self, img_or_mask_path, patch_index):
        """Extract a patch from an image or a mask given its index."""
        image = Image.open(img_or_mask_path)
        width, height = image.size
        step_size = self.patch_size - self.overlap
        count_x = math.ceil((width - self.overlap) / step_size)

        # Compute row and column indices
        patch_row = patch_index // count_x
        patch_col = patch_index % count_x

        left = patch_col * step_size
        top = patch_row * step_size
        right = min(left + self.patch_size, width)
        bottom = min(top + self.patch_size, height)

        patch = image.crop((left, top, right, bottom))
        return patch

    def __getitem__(self, idx):
        # image_and_mask_index is the index of the image and mask in the dataset (same index for both)
        image_and_mask_index, patch_index = self.index_map[idx]
        image_patch = self.get_patch(self.image_paths[image_and_mask_index], patch_index)
        if not self.inference_mode:
            mask_patch = self.get_patch(self.mask_paths[image_and_mask_index], patch_index)
            mask_patch = Mask(mask_patch)
            
        if self.transform is not None:
            if not self.inference_mode:
                image_patch, mask_patch = self.transform(image_patch, mask_patch)
                prompt = np.expand_dims(self.get_points(mask_patch, num_points=self.num_points), axis=1)
                print("Image shape: " + str(image_patch.shape))
                print("Mask shape: " + str(mask_patch.shape))
            else:
                image_patch = self.transform(image_patch)
                print("Image shape: " + str(image_patch.shape))

        if not self.inference_mode:
            num_masks = self.num_points
            return image_patch, mask_patch, prompt, num_masks
        else:
            return image_patch