from torchvision import transforms


def get_transformation(mean, std):
    """
    Returns training and testing transformations with specified mean and std.

    Args:
        mean (tuple): Tuple of means for each channel (R,G,B)
        std (tuple): Tuple of standard deviations for each channel (R,G,B)

    Returns:
        tuple: (train_transform, test_transform)
    """
    assert isinstance(mean, tuple) and len(mean) == 3, "mean must be a tuple of size 3"
    assert isinstance(std, tuple) and len(std) == 3, "std must be a tuple of size 3"
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
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, test_transform
