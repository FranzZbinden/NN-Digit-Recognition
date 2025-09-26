import os
import re
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import List, Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk, ImageDraw
import importlib
from dotenv import load_dotenv


MODEL_PATH = os.path.join("models", "trained", "handwritten.model.keras")
DIGITS_DIR = "digits"
FILENAME_PREFIX = "digit"
DISPLAY_SCALE = 10
DISPLAY_SIZE = (28 * DISPLAY_SCALE, 28 * DISPLAY_SCALE)
load_dotenv()

MODEL_MODULE = os.getenv("MODEL_MODULE", "models.trainingModels.trainingModelTwo")
_model = importlib.import_module(MODEL_MODULE)


# Preprocess a digit image for model prediction.
def preprocess_digit_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    batch = np.invert(np.array([img]))
    return batch


# Load and scale an image for display in the UI.
def load_image_for_display(image_path: str) -> ImageTk.PhotoImage:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    pil_img = Image.fromarray(img).convert("L")
    pil_img = pil_img.resize(DISPLAY_SIZE, Image.NEAREST)
    return ImageTk.PhotoImage(pil_img)


# Create a new PIL image and drawing object for the canvas.
def create_drawing_image(logical_size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.Draw]:
    image = Image.new("L", logical_size, color=255)
    draw = ImageDraw.Draw(image)
    return image, draw


# Save an image to the digits directory with the next available filename.
def save_image_to_digits_dir(image: Image.Image, digits_dir: str, filename_prefix: str) -> str:
    os.makedirs(digits_dir, exist_ok=True)
    
    existing = collect_digit_images(digits_dir, filename_prefix)
    used_numbers = set()
    pattern = re.compile(rf"^{re.escape(filename_prefix)}(\d+)\.png$", re.IGNORECASE)
    
    for path in existing:
        name = os.path.basename(path)
        m = pattern.match(name)
        if m:
            used_numbers.add(int(m.group(1)))
    
    next_idx = (max(used_numbers) + 1) if used_numbers else 1
    save_path = os.path.join(digits_dir, f"{filename_prefix}{next_idx}.png")
    
    image.save(save_path)
    return save_path


# Collect and sort digit image files from a directory.
def collect_digit_images(directory: str, prefix: str) -> List[str]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.png$", re.IGNORECASE)
    candidates: List[tuple[int, str]] = []
    
    try:
        for name in os.listdir(directory):
            match = pattern.match(name)
            if match:
                num = int(match.group(1))
                candidates.append((num, os.path.join(directory, name)))
    except FileNotFoundError:
        return []
    
    candidates.sort(key=lambda x: x[0])
    return [path for _, path in candidates]


# Remove temporary files safely.
def cleanup_temporary_files(file_paths: List[str]) -> None:
    for path in file_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass


# Ensure the model directory exists.
def ensure_model_directory_exists(model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)


# Load a trained Keras model from disk.
def load_trained_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


# Predict a digit from an image using the trained model.
def predict_digit(model: tf.keras.Model, image_path: str) -> int:
    batch = _model.preprocess_digit_image(image_path)
    preds = model.predict(batch, verbose=0)
    return int(np.argmax(preds))


# Train a new model with optional progress callback.
def train_model_with_progress(progress_callback=None) -> tf.keras.Model:
    x_train, y_train, x_test, y_test = _model.load_mnist_data()
    x_train_norm, x_test_norm = _model.normalize_images(x_train, x_test)
    
    # Build model
    model = _model.build_classification_model(
        input_shape=(28, 28), 
        hidden_units=128, 
        num_classes=10
    )
    
    # Prepare callbacks
    callbacks = []
    if progress_callback:
        callbacks.append(progress_callback)
    
    # Train model
    model.fit(x_train_norm, y_train, epochs=3, callbacks=callbacks, verbose=0)
    
    return model


# Save a trained model to disk.
def save_trained_model(model: tf.keras.Model, model_path: str) -> None:
    ensure_model_directory_exists(model_path)
    _model.save_model_to_disk(model, model_path)


# Configure the window theme and appearance.
def setup_window_theme(root: tk.Tk) -> None:
    try:
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except Exception:
        pass


# Create a drawing canvas with border and return canvas and its container.
def create_drawing_canvas(parent: tk.Widget, display_size: tuple[int, int]) -> tuple[tk.Canvas, tk.Frame]:
    border_px = 2
    outer_w = display_size[0] + border_px * 2
    outer_h = display_size[1] + border_px * 2
    
    board_border = tk.Frame(
        parent,
        width=outer_w,
        height=outer_h,
        background="#CCCCCC",
    )
    board_border.grid_propagate(False)
    
    inner_holder = tk.Frame(
        board_border,
        width=display_size[0],
        height=display_size[1],
        background="#FFFFFF",
    )
    inner_holder.grid(row=0, column=0, padx=border_px, pady=border_px)
    inner_holder.grid_propagate(False)
    
    canvas = tk.Canvas(
        inner_holder,
        width=display_size[0],
        height=display_size[1],
        bg="#FFFFFF",
        highlightthickness=0,
    )
    canvas.grid(row=0, column=0)
    
    return canvas, board_border


# Paint a single cell on the canvas.
def paint_canvas_cell(canvas: tk.Canvas, cx: int, cy: int, scale: int, color: str = "#000000") -> None:
    x0, y0 = cx * scale, cy * scale
    x1, y1 = x0 + scale, y0 + scale
    canvas.create_rectangle(x0, y0, x1, y1, outline=color, fill=color)


# Paint a line between two points on the canvas.
def paint_canvas_line(canvas: tk.Canvas, start: tuple[int, int], end: tuple[int, int], 
                     scale: int, logical_size: tuple[int, int], color: str = "#000000") -> None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    
    if steps == 0:
        paint_canvas_cell(canvas, x0, y0, scale, color)
        return
    
    for i in range(steps + 1):
        t = i / steps
        cx = int(round(x0 + dx * t))
        cy = int(round(y0 + dy * t))
        if 0 <= cx < logical_size[0] and 0 <= cy < logical_size[1]:
            paint_canvas_cell(canvas, cx, cy, scale, color)


# Convert screen coordinates to logical grid coordinates.
def screen_to_logical_coords(event_x: int, event_y: int, scale: int, logical_size: tuple[int, int]) -> tuple[int, int]:
    cx = max(0, min(logical_size[0] - 1, event_x // scale))
    cy = max(0, min(logical_size[1] - 1, event_y // scale))
    return cx, cy


class DigitPredictorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        container = ttk.Frame(root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        top_frame = ttk.Frame(container)
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=0)
        top_frame.columnconfigure(2, weight=1)
        top_frame.rowconfigure(0, weight=1)
        top_frame.rowconfigure(1, weight=0)
        top_frame.rowconfigure(2, weight=1)

        self.draw_scale = DISPLAY_SCALE
        self.draw_logical_size = (28, 28)
        center_holder = ttk.Frame(top_frame)
        center_holder.grid(row=1, column=1)
        
        self.draw_canvas, board_border = create_drawing_canvas(center_holder, DISPLAY_SIZE)
        board_border.grid(row=0, column=0)
        
        self._bind_drawing_events()
        self.draw_image, self.draw_draw = create_drawing_image(self.draw_logical_size)

        self.prediction_var = tk.StringVar(value="Draw a number between 1-9")
        self.prediction_label = ttk.Label(container, textvariable=self.prediction_var, font=("Segoe UI", 14))
        self.prediction_label.grid(row=1, column=0, pady=(8, 12))

        self.train_progress: tk.DoubleVar = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(container, orient="horizontal", mode="determinate", maximum=100, variable=self.train_progress)
        self.progress_bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        bottom_bar = ttk.Frame(container)
        bottom_bar.grid(row=3, column=0, sticky="ew")
        bottom_bar.columnconfigure(0, weight=1)
        bottom_bar.columnconfigure(1, weight=1)
        bottom_bar.columnconfigure(2, weight=1)

        self.train_button = ttk.Button(bottom_bar, text="Train Model", command=self.train_model_only)
        self.train_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.clear_button = ttk.Button(bottom_bar, text="Clear", command=self.clear_drawing_and_temp)
        self.clear_button.grid(row=0, column=1, sticky="ew", padx=6)

        self.quit_button = ttk.Button(bottom_bar, text="Quit", command=self.on_close)
        self.quit_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self._photo: Optional[ImageTk.PhotoImage] = None
        self.model: Optional[tf.keras.Model] = None
        self.image_paths: List[str] = []
        self.current_index: int = 0
        self.is_running: bool = False
        self.is_training: bool = False
        self.temp_drawn_paths: List[str] = []

    def start_predicting(self) -> None:
        if self.is_training:
            return
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Model not found", f"Could not find model at '{MODEL_PATH}'.\nRun training first to create it.")
            return

        if not os.path.isdir(DIGITS_DIR):
            messagebox.showerror("Images folder missing", f"Folder '{DIGITS_DIR}' does not exist.")
            return

        self.image_paths = collect_digit_images(DIGITS_DIR, FILENAME_PREFIX)
        if not self.image_paths:
            messagebox.showinfo("No images found", f"No images like '{FILENAME_PREFIX}#.png' found in '{DIGITS_DIR}'.")
            return

        try:
            if self.model is None:
                self.model = load_trained_model(MODEL_PATH)
        except Exception as exc:
            messagebox.showerror("Failed to load model", str(exc))
            return

        self.train_button.config(state=tk.DISABLED)
        self.is_running = True
        self.current_index = 0
        self.prediction_var.set("Prediction: —")
        self._predict_once()

    def _predict_once(self) -> None:
        if not self.is_running:
            return
        if not self.image_paths:
            self.prediction_var.set("No images to predict.")
            self.is_running = False
            self.train_button.config(state=tk.NORMAL)
            return
        image_path = self.image_paths[-1]
        try:
            assert self.model is not None
            predicted_digit = predict_digit(self.model, image_path)
            self.prediction_var.set(f"Prediction: {predicted_digit}")
        except Exception as exc:
            self.prediction_var.set(f"Error: {exc}")
        finally:
            self.is_running = False
            self.train_button.config(state=tk.NORMAL)

    def _bind_drawing_events(self) -> None:
        self._last_cell: Optional[tuple[int, int]] = None
        self.draw_color = "#000000"
        self.draw_canvas.bind("<Button-1>", self._on_draw_start)
        self.draw_canvas.bind("<B1-Motion>", self._on_draw_move)
        self.draw_canvas.bind("<ButtonRelease-1>", self._on_draw_end)

    def _on_draw_start(self, event) -> None:
        cell = screen_to_logical_coords(event.x, event.y, self.draw_scale, self.draw_logical_size)
        self._last_cell = cell
        self._paint_cell(cell[0], cell[1])

    def _on_draw_move(self, event) -> None:
        if self._last_cell is None:
            self._on_draw_start(event)
            return
        cell = screen_to_logical_coords(event.x, event.y, self.draw_scale, self.draw_logical_size)
        self._paint_line_cells(self._last_cell, cell)
        self._last_cell = cell

    def _on_draw_end(self, event) -> None:
        self._last_cell = None
        self._auto_save_drawing()
        if not self.is_training and not self.is_running:
            self.root.after(0, self.start_predicting)

    def _paint_cell(self, cx: int, cy: int) -> None:
        paint_canvas_cell(self.draw_canvas, cx, cy, self.draw_scale, self.draw_color)
        self.draw_image.putpixel((cx, cy), 0)

    def _paint_line_cells(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        paint_canvas_line(self.draw_canvas, start, end, self.draw_scale, self.draw_logical_size, self.draw_color)
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            self.draw_image.putpixel((x0, y0), 0)
            return
        for i in range(steps + 1):
            t = i / steps
            cx = int(round(x0 + dx * t))
            cy = int(round(y0 + dy * t))
            if 0 <= cx < self.draw_logical_size[0] and 0 <= cy < self.draw_logical_size[1]:
                self.draw_image.putpixel((cx, cy), 0)

    def save_drawing_to_digits(self) -> None:
        try:
            save_path = save_image_to_digits_dir(self.draw_image, DIGITS_DIR, FILENAME_PREFIX)
            self.temp_drawn_paths.append(save_path)
        except Exception as exc:
            self.prediction_var.set(f"Save error: {exc}")

    def _auto_save_drawing(self) -> None:
        try:
            prev_text = self.prediction_var.get()
            self.save_drawing_to_digits()
            if prev_text.startswith("Prediction:"):
                self.prediction_var.set(prev_text)
        except Exception as exc:
            self.prediction_var.set(f"Auto-save error: {exc}")

    def on_close(self) -> None:
        cleanup_temporary_files(self.temp_drawn_paths)
        self.root.destroy()

    def clear_drawing_and_temp(self) -> None:
        try:
            if self.temp_drawn_paths:
                last_path = self.temp_drawn_paths.pop()
                try:
                    if os.path.isfile(last_path):
                        os.remove(last_path)
                except Exception:
                    pass
            self.draw_canvas.delete("all")
            self.draw_image, self.draw_draw = create_drawing_image(self.draw_logical_size)
            self._last_cell = None
            self.prediction_var.set("Cleared.")
        except Exception as exc:
            self.prediction_var.set(f"Clear error: {exc}")

    def train_model_only(self) -> None:
        if self.is_running or self.is_training:
            return
        self.is_training = True
        self.prediction_var.set("Training model... 0%")
        self.train_progress.set(0.0)
        self.train_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._train_model_task, daemon=True)
        thread.start()

    def _train_model_task(self) -> None:
        try:
            class TkProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, app: "DigitPredictorApp") -> None:
                    super().__init__()
                    self.app = app
                    self.current_epoch = 0
                    self.total_epochs = 1
                    self.steps_per_epoch: int | None = None

                def on_train_begin(self, logs=None):
                    self.total_epochs = int(self.params.get("epochs", 1))
                    self.steps_per_epoch = self.params.get("steps", None)
                    self._update_percent(0.0)

                def on_epoch_begin(self, epoch, logs=None):
                    self.current_epoch = epoch

                def on_batch_end(self, batch, logs=None):
                    if self.steps_per_epoch:
                        completed = self.current_epoch * self.steps_per_epoch + (batch + 1)
                        total = self.total_epochs * self.steps_per_epoch
                        percent = max(0.0, min(100.0, (completed / total) * 100.0))
                        self._update_percent(percent)

                def on_epoch_end(self, epoch, logs=None):
                    percent = max(0.0, min(100.0, ((epoch + 1) / self.total_epochs) * 100.0))
                    self._update_percent(percent)

                def _update_percent(self, percent: float) -> None:
                    def do_update() -> None:
                        self.app.train_progress.set(percent)
                        self.app.prediction_var.set(f"Training model... {percent:.0f}%")
                    self.app.root.after(0, do_update)

            progress_callback = TkProgressCallback(self)
            model = train_model_with_progress(progress_callback)
            save_trained_model(model, MODEL_PATH)

            def on_done() -> None:
                self.model = model
                self.train_progress.set(100.0)
                self.prediction_var.set("Training complete. Model saved.")
                self.is_training = False
                self.train_button.config(state=tk.NORMAL)

            self.root.after(0, on_done)
        except Exception as exc:
            def on_error() -> None:
                self.prediction_var.set(f"Training error: {exc}")
                self.is_training = False
                self.train_button.config(state=tk.NORMAL)
            self.root.after(0, on_error)



def main() -> None:
    root = tk.Tk()
    root.geometry("500x500")
    
    setup_window_theme(root)
    
    app = DigitPredictorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()


