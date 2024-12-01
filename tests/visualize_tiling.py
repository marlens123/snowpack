import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from snowpack.augmentation import get_transformation, revert_normalization
from snowpack.dynamic_tiled_dataset import DynamicImagePatchesDataset

# Parameters
data_dir = "data_dir/test/"  # Update this path with either the path to the test or train set
image_mean = (0.5, 0.5, 0.5)  # Update these values
image_std = (0.5, 0.5, 0.5)  # Update these values
tile_size = 1024  # Experiment with different values
overlap = 128
batch_size = 32
batch_to_visualize = 0  # Change this value to visualize a different batch

# Transformations
train_transform, test_transform = get_transformation(mean=image_mean, std=image_std)

# Dataset and DataLoader minimal example of usage
dataset = DynamicImagePatchesDataset(data_dir, tile_size, overlap, transform=test_transform)
# shuffle=False to keep the order of the patches, typically for training we would want to shuffle the patches
# but for testing it can be useful to keep the order
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# these next few lines show some statistics about the dataset
print(f"Number of batches: {len(dataloader)}")
print(f"Number of patches: {len(dataset)}")
print(f"Number of images: {len(dataset.image_paths)}")
print(f"Number of patches per image: {dataset.image_patch_counts}")
print(f"Image paths: {dataset.image_paths}")

reference_image = np.array(Image.open(dataset.image_paths[0]))
reference_mask = np.array(Image.open(dataset.mask_paths[0]))

for batch_index, batch in enumerate(dataloader):
    # Visualize the first batch of 16 images and their corresponding masks in a 4x8 subplot
    images, masks = batch

    if batch_index == batch_to_visualize:
        _, axd = plt.subplot_mosaic(
            [
                ["1", "1", "A", "B", "C", "D", "E", "F", "AW", "AX"],
                ["1", "1", "G", "H", "I", "J", "K", "L", "AY", "AZ"],
                ["1", "1", "M", "N", "O", "P", "Q", "R", "BA", "BB"],
                ["1", "1", "S", "T", "U", "V", "W", "X", "BC", "BD"],
                ["2", "2", "Y", "Z", "AA", "AB", "AC", "AD", "BE", "BF"],
                ["2", "2", "AE", "AF", "AG", "AH", "AI", "AJ", "BG", "BH"],
                ["2", "2", "AK", "AL", "AM", "AN", "AO", "AP", "BI", "BJ"],
                ["2", "2", "AQ", "AR", "AS", "AT", "AU", "AV", "BK", "BL"],
            ],
            figsize=(20, 20),
        )

        # Display the original image and mask
        axd["1"].imshow(reference_image, cmap="gray")
        axd["1"].axis("off")
        axd["1"].set_title("Original Image", fontsize=30)
        axd["2"].imshow(reference_mask, cmap="viridis")
        axd["2"].axis("off")
        axd["2"].set_title("Original Mask", fontsize=30)

        # Display the patches and their corresponding masks
        patch_labels = [
            "A",
            "C",
            "E",
            "AW",
            "G",
            "I",
            "K",
            "AY",
            "M",
            "O",
            "Q",
            "BA",
            "S",
            "U",
            "W",
            "BC",
            "Y",
            "AA",
            "AC",
            "BE",
            "AE",
            "AG",
            "AI",
            "BG",
            "AK",
            "AM",
            "AO",
            "BI",
            "AQ",
            "AS",
            "AU",
            "BK",
        ]
        mask_labels = [
            "B",
            "D",
            "F",
            "AX",
            "H",
            "J",
            "L",
            "AZ",
            "N",
            "P",
            "R",
            "BB",
            "T",
            "V",
            "X",
            "BD",
            "Z",
            "AB",
            "AD",
            "BF",
            "AF",
            "AH",
            "AJ",
            "BH",
            "AL",
            "AN",
            "AP",
            "BJ",
            "AR",
            "AT",
            "AV",
            "BL",
        ]

        for i in range(32):
            axd[patch_labels[i]].imshow(
                revert_normalization(images[i], mean=image_mean, std=image_std).permute(1, 2, 0)
            )
            axd[patch_labels[i]].axis("off")

            axd[mask_labels[i]].imshow(masks[i].squeeze(), cmap="viridis")
            axd[mask_labels[i]].axis("off")

        plt.tight_layout()
        plt.show()
        break
