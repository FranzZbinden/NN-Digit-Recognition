# Model training and management logic.
# Handles model loading, training, prediction, and saving.

import os
import importlib
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv


MODEL_PATH = os.path.join("models", "trained", "handwritten.model.keras")

load_dotenv()
MODEL_MODULE = os.getenv("MODEL_MODULE", "models.trainingModels.trainingModelTwo")
_model = importlib.import_module(MODEL_MODULE)


def ensure_model_directory_exists(model_path: str) -> None:
    # Ensure the model directory exists.
    os.makedirs(os.path.dirname(model_path), exist_ok=True)


def load_trained_model(model_path: str) -> tf.keras.Model:
    # Load a trained Keras model from disk.
    return tf.keras.models.load_model(model_path)


def predict_digit(model: tf.keras.Model, image_path: str) -> int:
    # Predict a digit from an image using the trained model.
    batch = _model.preprocess_digit_image(image_path)
    preds = model.predict(batch, verbose=0)
    return int(np.argmax(preds))


def train_model_with_progress(progress_callback=None) -> tf.keras.Model:
    # Train a new model with optional progress callback.
    x_train, y_train, x_test, y_test = _model.load_mnist_data()
    x_train_norm, x_test_norm = _model.normalize_images(x_train, x_test)
    
    model = _model.build_classification_model(
        input_shape=(28, 28), 
        hidden_units=128, 
        num_classes=10
    )
    
    callbacks = []
    if progress_callback:
        callbacks.append(progress_callback)
    
    model.fit(x_train_norm, y_train, epochs=3, callbacks=callbacks, verbose=0)
    
    return model


def save_trained_model(model: tf.keras.Model, model_path: str) -> None:
    # Save a trained model to disk.
    ensure_model_directory_exists(model_path)
    _model.save_model_to_disk(model, model_path)

