import numpy as np
from sklearn.svm import LinearSVC


def svm_classify(
    train_image_feats: np.ndarray,
    train_labels: list[str],
    test_image_feats: np.ndarray,
    C: float = 0.1,
    class_weight: str | None = "balanced",
) -> list[str]:
    """Train a linear SVM classifier and predict test image categories.

    Uses one-vs-rest (OvR) multi-class strategy via sklearn's LinearSVC.

    Args:
        train_image_feats: (N, d) array of training features.
        train_labels: N-element list of ground-truth category strings.
        test_image_feats: (M, d) array of test features.
        C: Regularization parameter (smaller = stronger regularization).
        class_weight: 'balanced' adjusts for class frequency.

    Returns:
        M-element list of predicted category strings.
    """
    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        multi_class="ovr",
        class_weight=class_weight,
        random_state=0,
        tol=1e-5,
        max_iter=2000,
        dual=False,
    )
    clf.fit(train_image_feats, train_labels)
    return clf.predict(test_image_feats).tolist()
