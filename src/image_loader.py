"""Image loader module for the 15 scene dataset."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageLoader(Dataset):
    """PyTorch Dataset for loading images from the 15 scene dataset.

    This dataset loads grayscale images organized in class folders.
    Each class folder contains images for a specific scene category.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the ImageLoader dataset.

        Args:
            root_dir: The root directory of the dataset (e.g., "data/").
            split: Either "train" or "test" to select the data split.
            transform: Optional transform to be applied to the images.
        """
        self.transform = transform
        self.split = split
        self.root_dir = root_dir

        # Get all category folders
        split_path = Path(root_dir) / split
        cat_folders = sorted(split_path.glob("*"))

        # Create class dictionary mapping class names to indices
        self.class_dict: dict[str, int] = {}
        for idx, cat_path in enumerate(cat_folders):
            class_name = cat_path.name
            self.class_dict[class_name] = idx

        # Collect all image paths and their labels
        self.data: list[tuple[str, int]] = []
        for cat_path in cat_folders:
            class_name = cat_path.name
            label = self.class_dict[class_name]
            # Get all jpg images in the class folder
            for img_path in sorted(cat_path.glob("*.jpg")):
                self.data.append((str(img_path), label))

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The total number of images in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """Get the image and label at the given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing (image, label) where image is a PIL Image
            (with transform applied if provided) and label is an integer.
        """
        img_path, label = self.data[idx]
        img = self._load_image_from_path(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label

    def _load_image_from_path(self, path: str) -> Image.Image:
        """Load an image from the given path as a grayscale PIL Image.

        Args:
            path: The file path to the image.

        Returns:
            A grayscale PIL Image.
        """
        return Image.open(path).convert("L")
