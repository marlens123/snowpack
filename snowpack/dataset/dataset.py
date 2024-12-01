from torch.utils.data import Dataset as BaseDataset
from patchify import patchify
import numpy as np
import torch
import cv2

class SnowDataset(BaseDataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray, 
        transforms = None,
        size_strategy: str = 'resize_simple',
        mask_type: str = 'boundary',
        num_points: int = 50,
        resized_image_size: tuple = (1024, 1024),
    ):
        assert len(images) == len(masks)
        assert size_strategy in ['resize_simple', 'resize_retain_aspect', 'row_chunks', 'single_chunk']
        assert mask_type in ['boundary', 'boundary_revert', 'layer']

        self.transforms = transforms
        self.num_points = num_points

        self.size_strategy = size_strategy
        self.resized_image_size = resized_image_size
        self.mask_type = mask_type

        self.images = images
        self.masks = masks
        self.resized_images = []
        self.resized_masks = []

        # preprocess images and masks and store them in target_images and target_masks
        #for i in range(len(images)):
        #    self.change_size(images[i], masks[i], self.size_strategy)

        print("Number of images: " + str(len(self.images)))
        print("Number of masks: " + str(len(self.masks)))
    
    def change_size(self, image, mask):
        if "chunks" in self.size_strategy:
             # idea: during inference, patchify then unpatchify the image
             # for training, patchify the image and mask row wise to avoid too homogeneous patches
            chunked_image, chunked_mask = self.chunk_row_wise(image, mask)

            self.resized_images.append(chunked_image)
            self.resized_masks.append(chunked_mask)

        elif self.size_strategy == "resize":
            # problem with resizing: 2-times interpolation, which can lead to loss of information
            self.resized_images.append(cv2.resize(image, self.resized_image_size))
            # use nearest neighbour interpolation to make sure that masks remain categorical
            self.resized_masks.append(cv2.resize(mask, self.resized_image_size, interpolation=cv2.INTER_NEAREST))

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

    # TODO
    def get_bounding_box(binarized_mask):
        """
        Get bounding boxes for each unique label in the mask with added perturbation to the coordinates.
        
        Parameters:
        ground_truth_map (numpy array): The mask where non-zero values indicate the region of interest.
        
        Returns:
        bboxes (list of lists): A list of bounding box coordinates [x_min, y_min, x_max, y_max] for each label.
        """
        boundary_label = 1
        bboxes = []
        
        for label in boundary_label:
            indices = np.argwhere(binarized_mask == label)
            if len(indices) == 0:
                continue
            
            # Calculate bounding box coordinates
            y_min, x_min = np.min(indices[:, :2], axis=0)
            y_max, x_max = np.max(indices[:, :2], axis=0)
            
            bbox = [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
            bboxes = [[int(element) for element in sublist] for sublist in bboxes]
        return bboxes
    

    # TODO
    @staticmethod
    def binarize_single_object(original_mask):

        # Initialize a single binary mask
        binary_mask = np.zeros_like(original_mask, dtype=np.uint8)

        # Get binary masks and combine them into a single mask
        inds = np.unique(original_mask)[1:]  # Skip the background (index 0)
        for ind in inds:
            mask = (original_mask == ind).astype(np.uint8)  # Create binary mask for each unique index
            binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask

        return binary_mask


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

    @staticmethod
    def create_boundary_mask(mask, revert=False):
        # Perform edge detection in x and y direction
        sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate magnitude of gradient (boundary detection)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Threshold the gradient magnitude to create a binary boundary image
        _, boundary_mask = cv2.threshold(magnitude, 1, 255, cv2.THRESH_BINARY)

        # Convert to 0s and 1s
        binary_boundary = (boundary_mask > 0).astype(np.uint8)

        if revert:
            binary_boundary = 1 - binary_boundary

        assert np.unique(binary_boundary).tolist() == [0, 1]
        
        return binary_boundary

    # TODO
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
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = self.normalize_image(image)

        # TODO
        if self.transforms:
            image, mask = self.transform(image, mask)

        if self.mask_type == 'boundary':
            mask = self.create_boundary_mask(mask, revert=False)
            prompt = np.expand_dims(self.get_points(mask, num_points=self.num_points), axis=1)
        elif self.mask_type == 'boundary_revert':
            mask = self.create_boundary_mask(mask, revert=True)
            prompt = np.expand_dims(self.get_points(mask, num_points=self.num_points), axis=1)
        elif self.mask_type == 'layer':
            prompt = None
        
        image = self.expand_grayscale_channel(image)
        # image = image.permute(2, 0, 1)
        mask = np.expand_dims(mask, axis=0)
        num_masks = self.num_points

        # from: https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1_training/sam2.1_hiera_b%2B_MOSE_finetune.yaml
        # mean: [0.485, 0.456, 0.406]
        # std: [0.229, 0.224, 0.225]

        # apply transformations
        # TODO: check that interpolation=NEAREST NEIGHBOUR in rotation, and resizing!
        # TODO: check that resizing retains aspect ratio
        # TODO: does the prompt have to be in float?

        if self.mask_type == 'layer':
            return image.astype(np.float32), mask.astype(np.float32)
        else:
            return image.astype(np.float32), mask.astype(np.float32), prompt, num_masks
    
    # NEXT: check transforms - is interpolation nearest, check that resizing retains aspect ratio
    # check if grayscale function in the transforms works or if we should rather do it here
    # check how transforms handles images and masks

    # NEXT: retain aspect ratio