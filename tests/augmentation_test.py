import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from snowpack.augmentation import (
    get_transformation,
    normalize_image,
    revert_normalization,
)

# Global variable for image path
IMAGE_PATH = "assets/01_processed_20240108_16-29.tiff"  # Update this path
TRAIN_MEAN = (0.5, 0.5, 0.5)  # Update these values
TRAIN_STD = (0.5, 0.5, 0.5)  # Update these values


def show_images_grid(image_path, train_transform, test_transform, save_file_name=""):
    num_images = 16
    image = np.array(Image.open(image_path))
    # Transform the image and revert normalization to be able to plot it
    # The prmute call is there to change the order of the dimensions from (C, H, W) to (H, W, C)
    transformed_images = [
        revert_normalization(
            train_transform(normalize_image(image, expected_range=1)), mean=TRAIN_MEAN, std=TRAIN_STD
        ).permute(1, 2, 0)
        for _ in range(num_images)
    ]
    _, axd = plt.subplot_mosaic(
        [
            ["1", "1", "A", "B", "C", "D"],
            ["1", "1", "E", "F", "G", "H"],
            ["2", "2", "I", "J", "K", "L"],
            ["2", "2", "M", "N", "O", "P"],
        ],
        figsize=(12, 12),
    )

    original_image = np.array(Image.open(image_path))

    for i, (label, ax) in enumerate(axd.items()):
        if label == "1":
            ax.set_title("Original Image")
            ax.imshow(original_image, cmap="gray")
        elif label == "2":
            ax.set_title("Test Transform")
            ax.imshow(
                revert_normalization(
                    test_transform(normalize_image(image, expected_range=1)), mean=TRAIN_MEAN, std=TRAIN_STD
                ).permute(1, 2, 0)
            )
        else:
            ax.set_title("Train Transform")
            ax.imshow(transformed_images[i - 2])
        ax.axis("off")

    plt.tight_layout()
    plt.axis("off")
    if len(save_file_name) > 0:
        plt.savefig(save_file_name)
    plt.show()


train_transform, test_transform = get_transformation(TRAIN_MEAN, TRAIN_STD)
show_images_grid(IMAGE_PATH, train_transform, test_transform)
