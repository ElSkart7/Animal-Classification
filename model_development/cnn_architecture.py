import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape=(128, 128, 3), num_classes=10):
    """Build a baseline CNN for 10-class animal image classification."""
    inputs = layers.Input(shape=input_shape)

    # Block 1: 32 filters
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2: 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3: 128 filters
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    return model


if __name__ == "__main__":
    model = build_cnn(input_shape=(128, 128, 3), num_classes=10)
    model.summary()

    # Pass a single dummy RGB image (batch size = 1) to verify output shape and probabilities
    sample_image = np.random.rand(1, 128, 128, 3).astype(np.float32)
    predictions = model(sample_image, training=False).numpy()

    print("Sample output shape:", predictions.shape)  # Expected: (1, 10)
    print("Probabilities sum:", float(np.sum(predictions[0])))  # Expected: ~1.0
    print("Sample probabilities:", predictions[0])

