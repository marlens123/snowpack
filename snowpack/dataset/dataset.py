from torch.utils.data import Dataset as BaseDataset
import numpy as np
import cv2

class FullImageDataset(BaseDataset):
    def __init__(
        self,
        images,
        masks, 
        transforms = None,
        size_strategy: str = 'resize_simple',
        mask_type: str = 'boundary',
        num_points: int = 50,
        resized_image_size: tuple = (1024, 1024),
        dilate: bool = False,
    ):
        assert len(images) == len(masks)
        assert size_strategy in ['resize_simple', 'resize_retain_aspect']
        assert mask_type in ['boundary', 'boundary_revert', 'layer']

        self.transforms = transforms
        self.num_points = num_points

        self.size_strategy = size_strategy
        self.resized_image_size = resized_image_size
        self.mask_type = mask_type
        self.dilate = dilate

        self.images = images
        self.masks = masks

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

    @staticmethod
    def create_boundary_mask(mask, revert=False, dilate=False):
        # Perform edge detection in x and y direction
        sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate magnitude of gradient (boundary detection)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Threshold the gradient magnitude to create a binary boundary image
        _, boundary_mask = cv2.threshold(magnitude, 1, 255, cv2.THRESH_BINARY)

        # Convert to 0s and 1s
        binary_boundary = (boundary_mask > 0).astype(np.uint8)

        # Create a kernel for dilation. This kernel will define how much the boundary will expand.
        # You can change the kernel size to control how much you want to dilate the boundaries
        kernel_size = 20  # You can adjust this size (e.g., 3, 5, 7, etc.)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # A square kernel

        # Dilation operation
        if dilate:
            binary_boundary = cv2.dilate(binary_boundary, kernel, iterations=1)

        if revert:
            binary_boundary = 1 - binary_boundary

        assert np.unique(binary_boundary).tolist() == [0, 1]
        
        return binary_boundary

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
            mask = self.create_boundary_mask(mask, revert=False, dilate=self.dilate)
            prompt = np.expand_dims(self.get_points(mask, num_points=self.num_points), axis=1)
        elif self.mask_type == 'boundary_revert':
            mask = self.create_boundary_mask(mask, revert=True, dilate=self.dilate)
            prompt = np.expand_dims(self.get_points(mask, num_points=self.num_points), axis=1)
        elif self.mask_type == 'layer':
            prompt = None
        
        image = self.expand_grayscale_channel(image)
        mask = np.expand_dims(mask, axis=0)
        num_masks = self.num_points

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