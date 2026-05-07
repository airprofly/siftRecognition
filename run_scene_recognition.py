"""
Scene Recognition Pipeline

Three pipelines:
  1. Tiny Image features + k-Nearest Neighbor (k=3)
  2. Bag of SIFT (vocabulary via k-means) + k-Nearest Neighbor (k=15)
  3. Bag of SIFT + Linear SVM (C=0.1)

Usage:
    python run_scene_recognition.py
"""

from pathlib import Path

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from configs import APP_CONFIG
from datasets.scenedataset import SceneDataset
from models.kmeans import build_vocabulary
from models.recognition import (
    get_bags_of_sifts,
    get_tiny_images,
    nearest_neighbor_classify,
)
from models.svm import svm_classify
from utils.utils import rgb2gray, show_results


def load_data(
    root_dir: Path, num_per_cat: int | None = None
) -> tuple[list[np.ndarray], list[str]]:
    """Load grayscale images and string labels from a SceneDataset directory.

    Args:
        root_dir: Path to train/ or test/ subdirectory.
        num_per_cat: Limit images per category (None = all).

    Returns:
        (images, labels) where images is a list of grayscale [0,1] ndarrays
        and labels is a list of category name strings.
    """
    dataset = SceneDataset(root_dir=root_dir)

    images: list[np.ndarray] = []
    labels: list[str] = []
    counts: dict[str, int] = {}

    for i in range(len(dataset)):
        img_rgb, label_idx = dataset[i]
        cat_name = dataset.classes[label_idx]

        if num_per_cat is not None:
            count = counts.get(cat_name, 0)
            if count >= num_per_cat:
                continue
            counts[cat_name] = count + 1

        img_gray = rgb2gray(img_rgb.astype(np.float32) / 255.0)
        images.append(img_gray)
        labels.append(cat_name)

    return images, labels


def pipeline_tiny_images(
    train_images: list[np.ndarray],
    train_labels: list[str],
    test_images: list[np.ndarray],
    test_labels: list[str],
) -> list[str]:
    """Pipeline 1: Tiny image + k-NN."""
    k = APP_CONFIG.scene_rec.k_tiny_images
    logger.info(f"=== Pipeline 1: Tiny Images + k-NN (k={k}) ===")

    cfg = APP_CONFIG.scene_rec
    if cfg.tiny_train_full_path.exists() and cfg.tiny_test_full_path.exists():
        train_feats = np.load(str(cfg.tiny_train_full_path))
        test_feats = np.load(str(cfg.tiny_test_full_path))
        logger.info(f"Loaded cached tiny features from {cfg.tiny_train_full_path} and {cfg.tiny_test_full_path}")
    else:
        train_feats = get_tiny_images(train_images)
        test_feats = get_tiny_images(test_images)
        cfg.tiny_train_full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cfg.tiny_train_full_path), train_feats)
        np.save(str(cfg.tiny_test_full_path), test_feats)
        logger.info(f"Tiny features saved to {cfg.tiny_train_full_path} and {cfg.tiny_test_full_path}")
    logger.info(f"Train features: {train_feats.shape}, Test features: {test_feats.shape}")

    predicted = nearest_neighbor_classify(train_feats, train_labels, test_feats, k=k)
    accuracy = np.mean(np.array(predicted) == np.array(test_labels))
    logger.info(f"Accuracy: {accuracy:.2%}")

    show_results(train_labels, test_labels, cfg.categories, cfg.abbr_categories, predicted)
    return predicted


def _load_vocabulary(train_images: list[np.ndarray]) -> np.ndarray:
    """Load or build the visual vocabulary."""
    cfg = APP_CONFIG.scene_rec
    if cfg.vocab_full_path.exists():
        vocabulary = np.load(str(cfg.vocab_full_path))
        logger.info(f"Loaded vocabulary from {cfg.vocab_full_path}")
    else:
        logger.info(f"Building vocabulary (size={cfg.vocab_size}, stride={cfg.stride}) from training images ...")
        vocabulary = build_vocabulary(train_images, cfg.vocab_size, stride=cfg.stride)
        cfg.vocab_full_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cfg.vocab_full_path), vocabulary)
        logger.info(f"Vocabulary saved to {cfg.vocab_full_path}")
    return vocabulary


def _load_bof_features(
    images: list[np.ndarray], vocabulary: np.ndarray, feat_path: Path, tag: str
) -> np.ndarray:
    """Load or compute Bag-of-SIFT features for a set of images."""
    if feat_path.exists():
        feats = np.load(str(feat_path))
        logger.info(f"Loaded cached {tag} features from {feat_path}")
    else:
        feats = get_bags_of_sifts(images, vocabulary, stride=APP_CONFIG.scene_rec.stride)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(feat_path), feats)
        logger.info(f"{tag} features saved to {feat_path}")
    return feats


def pipeline_bag_of_sifts(
    train_images: list[np.ndarray],
    train_labels: list[str],
    test_images: list[np.ndarray],
    test_labels: list[str],
) -> list[str]:
    """Pipeline 2: Bag of SIFT + k-NN."""
    k = APP_CONFIG.scene_rec.k_bag_of_sifts
    logger.info(f"=== Pipeline 2: Bag of SIFT + k-NN (k={k}) ===")

    vocabulary = _load_vocabulary(train_images)
    cfg = APP_CONFIG.scene_rec

    logger.info("Extracting Bag of SIFT features for training set ...")
    train_feats = _load_bof_features(train_images, vocabulary, cfg.bof_train_full_path, "train")

    logger.info("Extracting Bag of SIFT features for test set ...")
    test_feats = _load_bof_features(test_images, vocabulary, cfg.bof_test_full_path, "test")

    logger.info(f"Train BoF features: {train_feats.shape}, Test BoF features: {test_feats.shape}")

    predicted = nearest_neighbor_classify(train_feats, train_labels, test_feats, k=k)
    accuracy = np.mean(np.array(predicted) == np.array(test_labels))
    logger.info(f"Accuracy: {accuracy:.2%}")

    show_results(train_labels, test_labels, cfg.categories, cfg.abbr_categories, predicted)
    return predicted


def pipeline_bag_of_sifts_svm(
    train_images: list[np.ndarray],
    train_labels: list[str],
    test_images: list[np.ndarray],
    test_labels: list[str],
) -> list[str]:
    """Pipeline 3: Bag of SIFT + Linear SVM."""
    cfg = APP_CONFIG.scene_rec
    logger.info(f"=== Pipeline 3: Bag of SIFT + Linear SVM (C={cfg.svm_c}) ===")

    vocabulary = _load_vocabulary(train_images)

    logger.info("Extracting Bag of SIFT features for training set ...")
    train_feats = _load_bof_features(train_images, vocabulary, cfg.bof_train_full_path, "train")

    logger.info("Extracting Bag of SIFT features for test set ...")
    test_feats = _load_bof_features(test_images, vocabulary, cfg.bof_test_full_path, "test")

    logger.info(f"Train BoF features: {train_feats.shape}, Test BoF features: {test_feats.shape}")

    predicted = svm_classify(train_feats, train_labels, test_feats, C=cfg.svm_c)
    accuracy = np.mean(np.array(predicted) == np.array(test_labels))
    logger.info(f"Accuracy: {accuracy:.2%}")

    show_results(train_labels, test_labels, cfg.categories, cfg.abbr_categories, predicted)
    return predicted


def main() -> None:
    logger.info("Loading training data ...")
    train_images, train_labels = load_data(APP_CONFIG.scene.train_dir, num_per_cat=APP_CONFIG.scene_rec.num_per_cat)
    logger.info(f"Loaded {len(train_images)} training images")

    logger.info("Loading test data ...")
    test_images, test_labels = load_data(APP_CONFIG.scene.test_dir)
    logger.info(f"Loaded {len(test_images)} test images")

    pipeline_tiny_images(train_images, train_labels, test_images, test_labels)

    pipeline_bag_of_sifts(train_images, train_labels, test_images, test_labels)

    pipeline_bag_of_sifts_svm(train_images, train_labels, test_images, test_labels)

    plt.show(block=True)


if __name__ == "__main__":
    main()
