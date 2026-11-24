
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def compile_model(model):
    model.compile(
        optimizer=Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Model compiled successfully!")
    print("Optimizer: Adam")
    print("Loss: Sparse Categorical Crossentropy")
    print("Metrics: Accuracy")

    return model
