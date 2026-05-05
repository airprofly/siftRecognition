from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


class SceneDataset(Dataset):
    """
    Scene recognition dataset organized by category subdirectories.

    Expects the directory structure:
        root_dir/
            category_1/
                image_0001.jpg
                image_0002.jpg
                ...
            category_2/
                image_0001.jpg
                ...

    Args:
        root_dir (str | Path): Directory containing category subdirectories, e.g.
            ``data/train`` or ``data/test``.
        transform (Callable[[np.ndarray], np.ndarray] | None): Callable applied to the
            loaded image array.

    Attributes:
        classes (list[str]): Sorted list of category names.
        class_to_idx (dict[str, int]): Mapping from category name to label index.
        samples (list[tuple[Path, int]]): List of ``(image_path, label)`` pairs.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        root = Path(root_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"\033[1;91mDataset root not found: {root}\033[0m"
            )

        self.transform = transform

        # Discover categories and build sample list in one pass
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.samples: list[tuple[Path, int]] = []

        for cat in sorted(root.iterdir(), key=lambda d: d.name):
            if not cat.is_dir():
                continue
            label = len(self.classes)
            self.classes.append(cat.name)
            self.class_to_idx[cat.name] = label
            for img_path in sorted(cat.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(
                f"\033[1;91mNo valid images found in: {root}\033[0m"
            )

        logger.info(
            f"SceneDataset [{root.name}]: {len(self.samples)} samples, {len(self.classes)} classes"
        )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Return the image array and label at the given index.

        Args:
            index (int): Sample index.

        Returns:
            tuple[np.ndarray, int]: ``(image, label)``.

        Raises:
            IndexError: If index is out of range.
        """
        if index >= len(self.samples) or index < 0:
            msg = (
                f"\033[1;91mIndex {index} out of range for dataset with "
                f"{len(self.samples)} samples\033[0m"
            )
            raise IndexError(msg)

        img_path, label = self.samples[index]
        image = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_scene_train_dataset(root_dir: Path) -> SceneDataset:
    """Build the training SceneDataset from the resolved training directory."""
    return SceneDataset(root_dir=root_dir)


def build_scene_test_dataset(root_dir: Path) -> SceneDataset:
    """Build the testing SceneDataset from the resolved testing directory."""
    return SceneDataset(root_dir=root_dir)


if __name__ == "__main__":
    # Example usage
    train_dataset = build_scene_train_dataset(Path("data/train"))
    test_dataset = build_scene_test_dataset(Path("data/test"))
    print(f"Train samples: {len(train_dataset)}, Classes: {train_dataset.classes}")