import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE: Tuple[int, int] = (128, 128)
NUM_CLASSES = 10
CLASS_NAMES: List[str] = [
    "Butterfly",
    "Cat",
    "Chicken",
    "Cow",
    "Dog",
    "Elephant",
    "Horse",
    "Sheep",
    "Spider",
    "Squirrel",
]

DATA_DIR = "prepared_data"
TEST_IMAGES_PATH = os.path.join(DATA_DIR, "test.npy")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.npy")

DEFAULT_MODEL_PATH = "animals10_transfer_model.keras"
OUTPUT_DIR = "evaluation_results"


def prepare_images(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def load_test_arrays(
    images_path: str = TEST_IMAGES_PATH,
    labels_path: str = TEST_LABELS_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    x_test = prepare_images(np.load(images_path))
    y_test = np.asarray(np.load(labels_path)).reshape(-1)
    return x_test, y_test


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}")
    return tf.keras.models.load_model(model_path)


def generate_predictions(
    model: tf.keras.Model,
    x: np.ndarray,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    x_prepared = prepare_images(x)
    probs = model.predict(x_prepared, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    return probs, y_pred


def map_indices_to_names(
    indices: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
) -> List[str]:
    return [class_names[int(idx)] for idx in indices]


def plot_confusion(
    cm: np.ndarray,
    names: Sequence[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Labeled Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def summarize_classification_errors(
    cm: np.ndarray,
    names: Sequence[str] = CLASS_NAMES,
    top_k: int = 5,
) -> str:
    confused_pairs = []
    num_classes = len(names)

    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((int(cm[i, j]), names[i], names[j]))

    confused_pairs.sort(key=lambda item: item[0], reverse=True)

    lines = ["=== Common Classification Errors Summary ==="]
    if not confused_pairs:
        lines.append("No misclassifications found.")
        return "\n".join(lines)

    lines.append(f"Top {min(top_k, len(confused_pairs))} confused pairs (True -> Predicted):")
    for count, true_name, pred_name in confused_pairs[:top_k]:
        lines.append(f"  - {true_name} -> {pred_name}: {count} samples")

    class_errors = []
    for i in range(num_classes):
        total_i = int(cm[i].sum())
        correct_i = int(cm[i, i])
        if total_i > 0:
            err_rate = (total_i - correct_i) / total_i
            class_errors.append((err_rate, total_i - correct_i, total_i, names[i]))
    class_errors.sort(key=lambda item: (item[0], item[1]), reverse=True)
    lines.append("\nHighest per-class error rates:")
    for err_rate, errors, total, name in class_errors[:3]:
        lines.append(f"  - {name}: {errors}/{total} misclassified ({err_rate:.1%})")

    return "\n".join(lines)


def evaluate_validation(
    model: tf.keras.Model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Tuple[float, float]:
    x_val_prep = prepare_images(x_val)
    y_val_arr = np.asarray(y_val).reshape(-1)

    val_loss, val_acc = model.evaluate(x_val_prep, y_val_arr, batch_size=batch_size, verbose=0)
    _, y_pred = generate_predictions(model, x_val_prep, batch_size=batch_size)

    labels = list(range(len(class_names)))
    print(f"[Validation] Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    print("\n[Validation] Classification Report:")
    print(
        classification_report(
            y_val_arr,
            y_pred,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
        )
    )
    return float(val_loss), float(val_acc)


def evaluate_final_test(
    model_path: str = DEFAULT_MODEL_PATH,
    images_path: str = TEST_IMAGES_PATH,
    labels_path: str = TEST_LABELS_PATH,
    batch_size: int = 32,
    class_names: Sequence[str] = CLASS_NAMES,
    output_dir: str = OUTPUT_DIR,
) -> Tuple[float, float, np.ndarray, str]:
    os.makedirs(output_dir, exist_ok=True)

    model = load_trained_model(model_path)
    x_test, y_test = load_test_arrays(images_path, labels_path)

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    _, y_pred = generate_predictions(model, x_test, batch_size=batch_size)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_path = os.path.join(output_dir, "confusion_matrix_test.png")
    plot_confusion(cm, names=class_names, save_path=cm_path)

    error_summary = summarize_classification_errors(cm, names=class_names)

    print(f"\n[Final Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    print("\n[Final Test] Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
        )
    )
    print(error_summary)

    summary_path = os.path.join(output_dir, "error_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(error_summary + "\n")

    return float(test_loss), float(test_acc), cm, error_summary


if __name__ == "__main__":
    evaluate_final_test()
