import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

DATA_DIR = "/content/drive/MyDrive/prepared_data"

import os
print(os.listdir(DATA_DIR))

train = np.load(os.path.join(DATA_DIR, "train.npy"), mmap_mode="r")

print("Train shape:", train.shape)
print("Data type:", train.dtype)

plt.figure(figsize=(5,5))
plt.imshow(train[0])
plt.axis("off")
plt.show()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
], name="data_augmentation")

sample = train[:4].astype("float32") / 255.0

augmented = data_augmentation(sample, training=True)

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(augmented[i])
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()
plt.show()

flip_layer = tf.keras.layers.RandomFlip("horizontal")

flipped = flip_layer(sample, training=True)

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(flipped[i])
    plt.title("Horizontal Flip")
    plt.axis("off")

plt.tight_layout()
plt.show()

rotation_layer = tf.keras.layers.RandomRotation(0.05)

rotated = rotation_layer(sample, training=True)

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(rotated[i])
    plt.title("Rotation")
    plt.axis("off")

plt.tight_layout()
plt.show()

zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=0.10,
    width_factor=0.10
)

zoomed = zoom_layer(sample, training=True)

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(zoomed[i])
    plt.title("Zoom")
    plt.axis("off")

plt.tight_layout()
plt.show()

translation_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.05,
    width_factor=0.05
)

translated = translation_layer(sample, training=True)

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(translated[i])
    plt.title("Translation")
    plt.axis("off")

plt.tight_layout()
plt.show()

test_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomTranslation(0.15, 0.15)
])

test_result = test_augmentation(sample, training=True)

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(test_result[i])
    plt.title("Stronger Augmentation")
    plt.axis("off")

plt.tight_layout()
plt.show()

RANDOM_STATE = 42

try:
    from tensorflow.keras.utils import Sequence as _BaseSequence
except ImportError:
    _BaseSequence = object


class SplitGenerator(_BaseSequence):

    def __init__(self, images_path, labels_path, batch_size=32, shuffle=False):
        self.images = np.load(images_path, mmap_mode="r")
        self.labels = np.load(labels_path)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = np.arange(len(self.labels))

        self.rng = np.random.default_rng(RANDOM_STATE)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        batch_ids = self.order[start:start + self.batch_size]

        batch_x = np.asarray(
            self.images[batch_ids], dtype=np.float32
        ) / 255.0

        batch_y = self.labels[batch_ids]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


class AugmentedTrainGenerator(SplitGenerator):

    def __getitem__(self, batch_index):
        batch_x, batch_y = super().__getitem__(batch_index)

        batch_x = self.augmentation(
            batch_x,
            training=True
        ).numpy()

        return batch_x, batch_y

    def __init__(self, *args, augmentation, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = augmentation


BATCH_SIZE = 32

train_aug_gen = AugmentedTrainGenerator(
    os.path.join(DATA_DIR, "train.npy"),
    os.path.join(DATA_DIR, "train_labels.npy"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    augmentation=data_augmentation
)

images, labels = train_aug_gen[0]

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Image range:", images.min(), images.max())
print("Labels:", labels[:10])

val_gen = SplitGenerator(
    os.path.join(DATA_DIR, "validation.npy"),
    os.path.join(DATA_DIR, "validation_labels.npy"),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_gen = SplitGenerator(
    os.path.join(DATA_DIR, "test.npy"),
    os.path.join(DATA_DIR, "test_labels.npy"),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation samples:", len(val_gen.labels))
print("Testing samples:", len(test_gen.labels))
print("Validation/Test augmentation: Disabled")

import matplotlib.pyplot as plt

original_images = images[:4]
augmented_images = data_augmentation(
    original_images,
    training=True
).numpy()

plt.figure(figsize=(12, 6))

for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(original_images[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.imshow(augmented_images[i])
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()
plt.savefig("augmentation_examples.png", dpi=300, bbox_inches="tight")
plt.show()
