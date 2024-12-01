import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from snowpack.augmentation import get_transformation, revert_normalization
from snowpack.dynamic_tiled_dataset import DynamicImagePatchesDataset

# Parameters
data_dir = "data_dir/test/"  # Update this path with either the path to the test or train set
image_mean = (0.5, 0.5, 0.5)  # Update these values
image_std = (0.5, 0.5, 0.5)  # Update these values
tile_size = 1024  # Experiment with different values
overlap = 128
batch_size = 16
batch_to_visualize = 1  # Change this value to visualize a different batch

# Transformations
train_transform, test_transform = get_transformation(mean=image_mean, std=image_std)

# Dataset and DataLoader minimal example of usage
dataset = DynamicImagePatchesDataset(data_dir, tile_size, overlap, transform=train_transform)
# shuffle=False to keep the order of the patches, typically for training we would want to shuffle the patches
# but for testing it can be useful to keep the order
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# these next few lines show some statistics about the dataset
print(f"Number of batches: {len(dataloader)}")
print(f"Number of patches: {len(dataset)}")
print(f"Number of images: {len(dataset.image_paths)}")
print(f"Number of patches per image: {dataset.image_patch_counts}")
print(f"Image paths: {dataset.image_paths}")


for batch_index, batch in enumerate(dataloader):
    # Visualize the first batch of 16 images and their corresponding masks in a 4x8 subplot
    images, masks = batch

    if batch_index == batch_to_visualize:
        fig, ax = plt.subplots(4, 8, figsize=(24, 12))
        for i in range(16):
            ax[i // 4, (i % 4) * 2].imshow(
                revert_normalization(images[i], mean=image_mean, std=image_std).permute(1, 2, 0)
            )  # permute is needed to change the order of the dimensions from (C, H, W) to (H, W, C)
            # revert_normalization is needed to revert the normalization applied to the image to bring the
            # pixel values back to the range [0, 1] as opposed to [-1, 1]
            ax[i // 4, (i % 4) * 2].axis("off")

            ax[i // 4, (i % 4) * 2 + 1].imshow(masks[i].squeeze(), cmap="viridis")
            ax[i // 4, (i % 4) * 2 + 1].axis("off")
        plt.tight_layout()
        plt.show()
        break
