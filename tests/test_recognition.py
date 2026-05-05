import math

import numpy as np

from models.recognition import pairwise_distances


def test_pairwise_distances():
    """
    Testing pairwise_distances()
    """
    # Test case 1
    actual_distances = np.array([[0, math.sqrt(2), 2]])

    X = np.array([[1, 1, 1, 1]])
    Y = np.array([[1, 1, 1, 1], [1, 2, 1, 2], [2, 2, 2, 2]])

    test_distances = pairwise_distances(X, Y)

    # Test case 2
    actual_distances_1 = np.array(
        [
            [math.sqrt(2), 1.0, math.sqrt(5)],
            [math.sqrt(3), 2.0, math.sqrt(2)],
            [math.sqrt(2), math.sqrt(5), math.sqrt(3)],
        ]
    )
    X_1 = np.array([[1, 1, 2], [2, 1, 0], [0, 1, 1]])
    Y_1 = np.array([[1, 2, 1], [2, 1, 2], [1, 0, 0]])

    test_distances_1 = pairwise_distances(X_1, Y_1)

    assert np.array_equal(actual_distances, test_distances)
    assert np.array_equal(actual_distances_1, test_distances_1)
