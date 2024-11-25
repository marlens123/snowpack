from torchvision import transforms


def get_transformation(mean, std):
    """
    Returns training and testing transformations with specified mean and std for grayscale images.

    Args:
        mean: Mean for the channels
        std: Standard deviation for the channels

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Crop a random portion of image and resize it to a given size.
            transforms.RandomResizedCrop(size=1024, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((1024, 1024)),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    return train_transform, test_transform
