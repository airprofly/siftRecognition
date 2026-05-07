from pathlib import Path

import numpy as np

from datasets.scenedataset import SceneDataset
from models.recognition import get_tiny_images


def build_tiny_images_dataloader(root_dir: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a tiny images feature matrix and labels from a SceneDataset directory.

    Args:
        root_dir (str | Path): Root directory containing category subdirectories.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str]]: ``(X, y, classes)`` where
        - X: (N, 256) feature matrix of normalized tiny images
        - y: (N,) integer label array
        - classes: sorted list of category names
    """
    dataset = SceneDataset(root_dir=root_dir)
    images = [dataset[i][0] for i in range(len(dataset))]
    labels = np.array([dataset.samples[i][1] for i in range(len(dataset))], dtype=np.int64)
    X = get_tiny_images(images)
    return X, labels, dataset.classes
