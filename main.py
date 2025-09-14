import os
from typing import Tuple

import importlib
import os

MODEL_MODULE = os.getenv("MODEL_MODULE", "models.trainingModels.trainingModelOne")
_model = importlib.import_module(MODEL_MODULE)

# Load the MNIST dataset and return train and test splits.
# main.py now delegates to modelOne for all model logic

# Main function to run the program.
def main() -> None:
    x_train, y_train, x_test, y_test = _model.load_mnist_data()
    x_train, x_test = _model.normalize_images(x_train, x_test)

    model = _model.build_classification_model(input_shape=(28, 28), hidden_units=128, num_classes=10)
    _model.train_model(model, x_train, y_train, epochs=3)

    model_dir = os.path.join("models", "trained")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "handwritten.model.keras")
    _model.save_model_to_disk(model, model_path)

    loaded_model = _model.load_model_from_disk(model_path)
    _model.predict_digits_in_directory(loaded_model, directory="digits", filename_prefix="digit")

    print("Program finished running.")


if __name__ == "__main__":
    main()
