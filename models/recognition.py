import numpy as np


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
    # X[:, np.newaxis, :] -> shape (N, 1, d)
    # Y[np.newaxis, :, :] -> shape (1, M, d)
    # broadcast difference -> shape (N, M, d)
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    D = np.linalg.norm(diff, axis=-1)
    return D


def get_tiny_images(image_path: str) -> np.ndarray:
    """
    Load a single image from path, resize to 16x16 grayscale, flatten and normalize.

    The tiny image feature is inspired by:
        80 million tiny images: a large dataset for non-parametric object and
        scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
        Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
        pp. 1958-1970, 2008.

    The image is resized to 16x16 (ignoring aspect ratio), flattened, then
    normalized to zero mean and unit length.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: 1D array of length 256 (16x16) representing the normalized
        tiny image feature.
    """
    from PIL import Image

    image = Image.open(image_path).convert("L")
    resized = image.resize((16, 16), Image.Resampling.BILINEAR)
    feat = np.asarray(resized, dtype=np.float64).flatten()

    # Zero mean and unit length normalization
    mean = np.mean(feat)
    std = np.std(feat)
    if std > 0:
        feat = (feat - mean) / std
    else:
        feat = feat - mean

    return feat
