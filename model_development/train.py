
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

INPUT_SHAPE = (128, 128, 3)
NUM_CLASSES = 10
EPOCHS = 10

MODEL_PATH = Path(
    "/kaggle/working/models/animals10_baseline_cnn.keras"
)


def train_model(train_dataset, val_dataset, epochs=EPOCHS):

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Build CNN
    model = build_cnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES
    )

    # Compile model
    model = compile_model(model)

    # Save the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )

    # Stop if validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            checkpoint,
            early_stopping
        ]
    )

    best_epoch = (
        min(
            range(len(history.history["val_loss"])),
            key=history.history["val_loss"].__getitem__
        ) + 1
    )

    print(f"\nBest epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{min(history.history['val_loss']):.4f}"
    )
    print(f"Saved model: {MODEL_PATH}")

    return model, history


def plot_training_history(history):

    epoch_numbers = range(
        1, len(history.history["loss"]) + 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epoch_numbers,
        history.history["loss"],
        label="Training"
    )
    axes[0].plot(
        epoch_numbers,
        history.history["val_loss"],
        label="Validation"
    )
    axes[0].set_title("Loss")

    axes[1].plot(
        epoch_numbers,
        history.history["accuracy"],
        label="Training"
    )
    axes[1].plot(
        epoch_numbers,
        history.history["val_accuracy"],
        label="Validation"
    )
    axes[1].set_title("Accuracy")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.show()

