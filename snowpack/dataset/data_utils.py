import rasterio
import os
import cv2
import numpy as np

def load_tifs_resize_to_np(image_path: str, mask_path: str, size: int = (1024, 1024)):

    images = []
    masks = []

    for _, i in enumerate(os.listdir(image_path)):
        image = rasterio.open(image_path + i).read(1)
        images.append(cv2.resize(image, size))

    for _, i in enumerate(os.listdir(mask_path)):
        mask= rasterio.open(mask_path + i).read(1)
        # use nearest neighbour interpolation to make sure that masks remain categorical
        masks.append(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))

    image_np = np.array(images)
    mask_np = np.array(masks)

    return image_np, mask_np


def load_tifs_resize_to_np_retain_ratio(image_path: str, mask_path: str, size: int = (1024, 1024)):

    images = []
    masks = []

    for _, i in enumerate(os.listdir(image_path)):
        image = rasterio.open(image_path + i).read(1)
        r = np.min([size[0] / image.shape[1], size[0] / image.shape[0]])
        images.append(cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r))))

    for _, i in enumerate(os.listdir(mask_path)):
        mask= rasterio.open(mask_path + i).read(1)
        r = np.min([size[0] / mask.shape[1], size[0] / mask.shape[0]])
        # use nearest neighbour interpolation to make sure that masks remain categorical
        masks.append(cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST))

    return images, masks