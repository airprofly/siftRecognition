from typing import Tuple

import numpy as np
from torchvision import transforms


def get_fundamental_transforms(
    inp_size: Tuple[int, int], pixel_mean: np.ndarray, pixel_std: np.ndarray
) -> transforms.Compose:
    """
    Returns the core transforms needed to feed the images to our model

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean  of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - fundamental_transforms: transforms.Compose with the fundamental transforms
    """

    fundamental_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std),
    ])
    return fundamental_transforms


def get_data_augmentation_transforms(
    inp_size: Tuple[int, int], pixel_mean: np.ndarray, pixel_std: np.ndarray
) -> transforms.Compose:
    """
    Returns the data augmentation + core transforms needed to be applied on the
    train set

    Args:
    - inp_size: tuple denoting the dimensions for input to the model
    - pixel_mean: the mean  of the raw dataset
    - pixel_std: the standard deviation of the raw dataset
    Returns:
    - aug_transforms: transforms.Compose with all the transforms
    """

    aug_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std),
    ])
    return aug_transforms
