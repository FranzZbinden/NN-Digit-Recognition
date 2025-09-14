import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def normalize_images(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize to [0,1] and add channel dimension for CNN input."""
    x_train_norm = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test_norm = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return x_train_norm, x_test_norm


def build_classification_model(
    input_shape: Tuple[int, int] = (28, 28),
    hidden_units: int = 128,
    num_classes: int = 10,
) -> tf.keras.Model:
    """Create and compile a CNN classifier for better digit recognition."""
    # Ensure channel dimension is included
    if len(input_shape) == 2:
        input_shape = (input_shape[0], input_shape[1], 1)

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 3,
) -> tf.keras.callbacks.History:
    """Train with sensible defaults and callbacks for better generalization."""
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=0),
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_model_to_disk(model: tf.keras.Model, filepath: str) -> None:
    model.save(filepath)


def load_model_from_disk(filepath: str) -> tf.keras.Model:
    return tf.keras.models.load_model(filepath)


def preprocess_digit_image(image_path: str) -> np.ndarray:
    """Read, invert like original, normalize to [0,1], add channel dim, and batch."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    inv = np.invert(img).astype("float32") / 255.0
    inv = inv[..., np.newaxis]  # (28,28,1)
    batch = np.expand_dims(inv, axis=0)  # (1,28,28,1)
    return batch


def predict_digits_in_directory(
    model: tf.keras.Model,
    directory: str = "digits",
    filename_prefix: str = "digit",
) -> None:
    image_number = 1
    while os.path.isfile(os.path.join(directory, f"{filename_prefix}{image_number}.png")):
        image_path = os.path.join(directory, f"{filename_prefix}{image_number}.png")
        try:
            img_batch = preprocess_digit_image(image_path)
            prediction = model.predict(img_batch)
            predicted_digit = int(np.argmax(prediction))
            print(f"This digit is probably a {predicted_digit}")
            plt.imshow(img_batch[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as exc:
            print(f"Error with {filename_prefix}{image_number}.png: {exc}")
        finally:
            image_number += 1
    print(f"Checking for file: {directory}/{filename_prefix}{image_number}.png")


