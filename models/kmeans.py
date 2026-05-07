import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from models.siftNet import get_siftnet_features


def kmeans(feature_vectors: np.ndarray, k: int, max_iter: int = 10) -> np.ndarray:
    """
    Implement the k-means algorithm in this function. Initialize your centroids
    with random *unique* points from the input data, and repeat over the
    following process:
    1. calculate the distances from data points to the centroids
    2. assign them labels based on the distance - these are the clusters
    3. re-compute the centroids from the labeled clusters

    Please note that you are NOT allowed to use any library functions like
    vq.kmeans from scipy or kmeans from vlfeat to do the computation!

    Useful functions:
    -   np.random.choice
    -   np.linalg.norm
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (N, d)
            where N is the number of features and d is the dimensionality of the
            features
    -   k: the number of centroids to generate, of type int
    -   max_iter: the total number of iterations for k-means to run, of type int

    Returns:
    -   centroids: the generated centroids for the input feature_vectors, a Numpy
            array of shape (k, d)
    """
    N, d = feature_vectors.shape

    # k-means++ initialization
    centroids = np.zeros((k, d))
    centroids[0] = feature_vectors[np.random.randint(N)]
    for i in range(1, k):
        diff = feature_vectors[:, np.newaxis, :] - centroids[:i][np.newaxis, :, :]
        dists = np.min(np.linalg.norm(diff, axis=-1), axis=1)
        # Add small epsilon to avoid division by zero
        dists = dists + 1e-10
        probs = dists / dists.sum()
        centroids[i] = feature_vectors[np.random.choice(N, p=probs)]

    for iteration in tqdm(range(max_iter), desc="k-means clustering", unit="iter"):
        # Compute pairwise distances (N, k) using broadcasting
        diff = feature_vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (N, k, d)
        distances = np.linalg.norm(diff, axis=-1)  # (N, k)

        # Assign each point to nearest centroid
        labels = np.argmin(distances, axis=1)  # (N,)

        # Recompute centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centroids[j] = feature_vectors[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]  # keep old centroid if empty cluster

        # Check convergence
        if np.allclose(centroids, new_centroids):
            tqdm.write(f"Converged at iteration {iteration + 1}")
            break
        centroids = new_centroids

        if iteration % 5 == 0:
            logger.info(f"k-means iteration {iteration + 1}/{max_iter}")

    return centroids


def build_vocabulary(image_arrays, vocab_size, stride=20):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Load images from the training set. To save computation time, you don't
    necessarily need to sample from all images, although it would be better
    to do so. You can randomly sample the descriptors from each image to save
    memory and speed up the clustering. For testing, you may experiment with
    larger stride so you just compute fewer points and check the result quickly.

    In order to pass the unit test, leave out a 10-pixel margin in the image,
    that is, start your x and y from 10, and stop at len(image_width) - 10 and
    len(image_height) - 10.

    For each loaded image, get some SIFT features. You don't have to get as
    many SIFT features as you will in get_bags_of_sifts, because you're only
    trying to get a representative sample here.

    Once you have tens of thousands of SIFT features from many training
    images, cluster them with kmeans. The resulting centroids are now your
    visual word vocabulary.

    Note that the default vocab_size of 50 is sufficient for you to get a decent
    accuracy (>40%), but you are free to experiment with other values.

    Useful functions:
    -   np.array(img, dtype='float32'), torch.from_numpy(img_array), and
            img_tensor = img_tensor.reshape(
                (1, 1, img_array.shape[0], img_array.shape[1]))
            for converting a numpy array to a torch tensor for siftnet
    -   get_siftnet_features() from SIFTNet: you can pass in the image tensor in
            grayscale, together with the sampled x and y positions to obtain the
            SIFT features
    -   np.arange() and np.meshgrid(): for you to generate the sample x and y
            positions faster

    Args:
    -   image_arrays: list of images in Numpy arrays, in grayscale
    -   vocab_size: size of vocabulary
    -   stride: the stride of your SIFT sampling

    Returns:
    -   vocab: This is a (vocab_size, dim) Numpy array (vocabulary). Where dim
            is the length of your SIFT descriptor. Each row is a cluster center
            / visual word.
    """

    all_descriptors = []
    log_interval = max(1, len(image_arrays) // 20)

    for idx, img in enumerate(tqdm(image_arrays, desc="Extracting SIFT features", unit="img")):
        img_float32 = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_float32).reshape(1, 1, img.shape[0], img.shape[1])

        h, w = img.shape[:2]
        ys = np.arange(10, h - 10, stride)
        xs = np.arange(10, w - 10, stride)
        yv, xv = np.meshgrid(ys, xs, indexing="ij")
        yv, xv = yv.flatten(), xv.flatten()

        if len(xv) == 0:
            continue

        fvs = get_siftnet_features(img_tensor, xv, yv)
        all_descriptors.append(fvs)

        if idx % log_interval == 0:
            logger.info(f"SIFT features [{idx}/{len(image_arrays)}]")

    if not all_descriptors:
        return np.zeros((vocab_size, 128))

    all_descriptors = np.vstack(all_descriptors)
    logger.info(f"Collected {all_descriptors.shape[0]} SIFT descriptors")

    # Remove NaN and inf values (can occur in flat image regions)
    valid_mask = np.isfinite(all_descriptors).all(axis=1)
    all_descriptors = all_descriptors[valid_mask]
    logger.info(f"Filtered to {all_descriptors.shape[0]} valid descriptors (removed NaN/inf)")

    if all_descriptors.shape[0] == 0:
        logger.warning("No valid descriptors found, returning zero vocabulary")
        return np.zeros((vocab_size, 128))

    # Subsample if too many descriptors (speed)
    if all_descriptors.shape[0] > 20000:
        idx = np.random.choice(all_descriptors.shape[0], 20000, replace=False)
        all_descriptors = all_descriptors[idx]
        logger.info(f"Subsampled to 20000 descriptors for clustering")

    logger.info(f"Running k-means (k={vocab_size}, max_iter=50) on {all_descriptors.shape[0]} descriptors ...")
    vocab = kmeans(all_descriptors, vocab_size, max_iter=50)
    logger.info("k-means clustering complete")
    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    Implement the k-means quantization in this function. Given the input data
    and the centroids, assign each of the data entry to the closest centroid.

    Useful functions:
    -   pairwise_distances
    -   np.argmin

    Args:
    -   raw_data_pts: the input data collection, a Numpy array of shape (N, d)
            where N is the number of input data, and d is the dimension of it,
            given the standard SIFT descriptor, d = 128
    -   centroids: the generated centroids for the input feature_vectors, a
            Numpy array of shape (k, D)

    Returns:
    -   indices: the index of the centroid which is closest to the data points,
            a Numpy array of shape (N, )

    """
    from models.recognition import pairwise_distances

    dists = pairwise_distances(raw_data_pts, centroids)
    indices = np.argmin(dists, axis=1)
    return indices
