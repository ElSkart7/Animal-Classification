import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(model_path, test_dataset, class_names):

    # Step 1: Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Step 2: Get the true labels
    y_true = []

    for images, labels in test_dataset:
        labels = labels.numpy()

        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        else:
            labels = labels.reshape(-1)

        y_true.extend(labels)

    y_true = np.array(y_true)

    # Step 3: Calculate Test Loss and Accuracy
    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print("Model Evaluation Results")
    print("------------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy Percentage: {test_accuracy * 100:.2f}%")

    # Step 4: Generate Predictions
    predictions = model.predict(test_dataset)

    y_pred = np.argmax(predictions, axis=1)

    # Step 5: Classification Report
    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0
        )
    )

    # Step 6: Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title("Animal Classification Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Step 7: Find the most confused animal classes
    confusion_pairs = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    (class_names[i], class_names[j], cm[i, j])
                )

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost Confused Animal Classes:")

    for actual, predicted, count in confusion_pairs[:5]:
        print(
            f"Actual: {actual} | Predicted: {predicted} | Count: {count}"
        )

    return test_loss, test_accuracy, cm
