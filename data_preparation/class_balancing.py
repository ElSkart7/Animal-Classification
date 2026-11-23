
import numpy as np
from collections import Counter


def calculate_class_weights(labels):
    """
    Calculate class weights based on class frequencies.
    """

    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)

    class_weights = {}

    for class_id, count in class_counts.items():
        class_weights[class_id] = (
            total_samples / (num_classes * count)
        )

    return class_weights

