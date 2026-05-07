import numpy as np
from tqdm import tqdm


def pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances between two sets of vectors.

    Args:
        X (np.ndarray): N x d numpy array of d-dimensional features arranged along N rows.
        Y (np.ndarray): M x d numpy array of d-dimensional features arranged along M rows.

    Returns:
        np.ndarray: N x M numpy array where d(i, j) is the distance between row i of X and
        row j of Y.
    """
    # Use (a-b)^2 = a^2 + b^2 - 2ab to avoid materializing the (N, M, d) tensor
    xx = np.sum(X**2, axis=1)                     # (N,)
    yy = np.sum(Y**2, axis=1)                     # (M,)
    D = np.sqrt(np.maximum(xx[:, np.newaxis] + yy[np.newaxis, :] - 2 * X @ Y.T, 0))
    return D


def _compute_tiny_image(image: np.ndarray) -> np.ndarray:
    """Resize a single grayscale image to 16x16, flatten and normalize."""
    from PIL import Image

    image_uint8 = image.astype(np.uint8)
    resized = Image.fromarray(image_uint8, mode="L").resize(
        (16, 16), Image.Resampling.BILINEAR
    )
    feat = np.asarray(resized, dtype=np.float64).flatten()

    mean = np.mean(feat)
    std = np.std(feat)
    if std > 0:
        feat = (feat - mean) / std
    else:
        feat = feat - mean
    return feat


def get_tiny_images(image_arrays: list[np.ndarray]) -> np.ndarray:
    """
    Extract tiny image features from a list of grayscale image arrays.

    Each image is resized to 16x16, flattened, then normalized to zero mean
    and unit length.

    Inspired by:
        80 million tiny images: a large dataset for non-parametric object and
        scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
        Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
        pp. 1958-1970, 2008.

    Args:
        image_arrays (list[np.ndarray]): List of N grayscale images as numpy arrays.

    Returns:
        np.ndarray: N x 256 feature matrix of normalized tiny images.
    """
    feats = np.empty((len(image_arrays), 256), dtype=np.float64)
    for i, image in enumerate(tqdm(image_arrays, desc="Tiny images")):
        feats[i] = _compute_tiny_image(image)
    return feats


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=3):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).
    Useful functions:
    -   D = pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y
    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    -   k: the k value in kNN, indicating how many votes we need to check for
            the label
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """

    D = pairwise_distances(test_image_feats, train_image_feats)
    test_labels = []

    for row in tqdm(D, desc="k-NN classify"):
        nearest_indices = np.argpartition(row, k)[:k]
        nearest_labels = [train_labels[i] for i in nearest_indices]
        test_labels.append(max(set(nearest_labels), key=nearest_labels.count))

    return test_labels


def get_bags_of_sifts(
    image_arrays: list[np.ndarray], vocabulary: np.ndarray, stride: int = 20
) -> np.ndarray:
    """
    Represent each image as a bag-of-features histogram of visual words.

    For each image, extract dense SIFT descriptors, quantize each descriptor
    to the nearest visual word in the vocabulary, and build a normalized
    histogram of visual word counts.

    Args:
        image_arrays (list[np.ndarray]): List of N grayscale images.
        vocabulary (np.ndarray): (vocab_size, 128) array of visual word centroids.
        stride (int): Sampling stride for dense SIFT extraction, default 20.

    Returns:
        np.ndarray: (N, vocab_size) normalized bag-of-features histograms.
    """
    from loguru import logger
    import torch

    from models.kmeans import kmeans_quantize
    from models.siftNet import get_siftnet_features

    vocab_size = vocabulary.shape[0]
    histograms = np.zeros((len(image_arrays), vocab_size))
    log_interval = max(1, len(image_arrays) // 20)  # ~5% 进度输出一次

    for i, img in enumerate(tqdm(image_arrays, desc="BoF features")):

        img_float32 = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_float32).reshape(
            1, 1, img.shape[0], img.shape[1]
        )

        h, w = img.shape[:2]
        ys = np.arange(10, h - 10, stride)
        xs = np.arange(10, w - 10, stride)
        yv, xv = np.meshgrid(ys, xs, indexing="ij")
        yv, xv = yv.flatten(), xv.flatten()

        if len(xv) == 0:
            continue

        fvs = get_siftnet_features(img_tensor, xv, yv)
        labels = kmeans_quantize(fvs, vocabulary)
        hist, _ = np.histogram(labels, bins=vocab_size, range=(0, vocab_size))
        hist = hist.astype(np.float64)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        histograms[i] = hist

        if i % log_interval == 0:
            logger.info(f"BoF features [{i}/{len(image_arrays)}]")

    return histograms
