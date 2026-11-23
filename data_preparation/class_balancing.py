
import numpy as np


def create_balanced_indices(labels, random_state=42):
    """
    Oversample minority classes until all classes
    have the same number of samples.
    """

    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)

    classes, counts = np.unique(labels, return_counts=True)
    target_count = counts.max()

    balanced_indices = []

    for class_id in classes:
        class_indices = np.where(labels == class_id)[0]

        sampled_indices = rng.choice(
            class_indices,
            size=target_count,
            replace=True
        )

        balanced_indices.extend(sampled_indices)

    balanced_indices = np.asarray(balanced_indices)
    rng.shuffle(balanced_indices)

    return balanced_indices
