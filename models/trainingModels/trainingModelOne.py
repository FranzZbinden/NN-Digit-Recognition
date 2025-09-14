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
    x_train_norm = tf.keras.utils.normalize(x_train, axis=1)
    x_test_norm = tf.keras.utils.normalize(x_test, axis=1)
    return x_train_norm, x_test_norm


def build_classification_model(
    input_shape: Tuple[int, int] = (28, 28),
    hidden_units: int = 128,
    num_classes: int = 10,
) -> tf.keras.Model:
    """Create and compile a simple feedforward classification model."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(hidden_units, activation="relu"))
    model.add(tf.keras.layers.Dense(hidden_units, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
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
    history = model.fit(x_train, y_train, epochs=epochs)
    return history


def save_model_to_disk(model: tf.keras.Model, filepath: str) -> None:
    model.save(filepath)


def load_model_from_disk(filepath: str) -> tf.keras.Model:
    return tf.keras.models.load_model(filepath)


def preprocess_digit_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    batch = np.invert(np.array([img]))
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


