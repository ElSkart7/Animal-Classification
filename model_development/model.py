
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def compile_model(model, train_dataset, num_classes=10):

    # Step 1: Check label shape
    for images, labels in train_dataset.take(1):

        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
        print("Labels dtype:", labels.dtype)

        # Step 2: Select loss function
        if len(labels.shape) == 1:
            loss_function = "sparse_categorical_crossentropy"

        elif len(labels.shape) == 2 and labels.shape[-1] == 1:
            loss_function = "sparse_categorical_crossentropy"

        elif len(labels.shape) == 2 and labels.shape[-1] == num_classes:
            loss_function = "categorical_crossentropy"

        else:
            raise ValueError("Unsupported label format")

    # Step 3: Compile model
    model.compile(
        optimizer=Adam(),
        loss=loss_function,
        metrics=["accuracy"]
    )

    print("Model compiled successfully!")
    print("Loss function:", loss_function)
    print("Optimizer: Adam")
    print("Metrics: accuracy")

    return model

