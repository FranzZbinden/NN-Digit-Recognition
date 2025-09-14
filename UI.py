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


MODEL_PATH = os.path.join("models", "trained", "handwritten.model.keras")
DIGITS_DIR = "digits"
FILENAME_PREFIX = "digit"
DISPLAY_SCALE = 10  # 28px -> 280px for visibility
DISPLAY_SIZE = (28 * DISPLAY_SCALE, 28 * DISPLAY_SCALE)

# Dynamic model module selection via environment variable
MODEL_MODULE = os.getenv("MODEL_MODULE", "models.trainingModels.trainingModelTwo")
_model = importlib.import_module(MODEL_MODULE)


def preprocess_digit_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Expecting 28x28 images; keep behavior identical to main.py (invert only)
    batch = np.invert(np.array([img]))
    return batch


def load_image_for_display(image_path: str) -> ImageTk.PhotoImage:
    # Kept for potential future preview use; not used in single-predict mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    pil_img = Image.fromarray(img).convert("L")
    pil_img = pil_img.resize(DISPLAY_SIZE, Image.NEAREST)
    return ImageTk.PhotoImage(pil_img)


class DigitPredictorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        # UI elements
        container = ttk.Frame(root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        # Top frame holds drawing canvas (left) and preview image (right)
        top_frame = ttk.Frame(container)
        top_frame.grid(row=0, column=0, sticky="nsew")
        # Center content using spacer rows/columns
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=0)
        top_frame.columnconfigure(2, weight=1)
        top_frame.rowconfigure(0, weight=1)
        top_frame.rowconfigure(1, weight=0)
        top_frame.rowconfigure(2, weight=1)

        # Drawing canvas (interactive) shown scaled to DISPLAY_SIZE, mapped to 28x28 logical pixels
        self.draw_scale = DISPLAY_SCALE
        self.draw_logical_size = (28, 28)
        center_holder = ttk.Frame(top_frame)
        center_holder.grid(row=1, column=1)
        # Grey border around the drawing board
        border_px = 2
        outer_w = DISPLAY_SIZE[0] + border_px * 2
        outer_h = DISPLAY_SIZE[1] + border_px * 2
        board_border = tk.Frame(
            center_holder,
            width=outer_w,
            height=outer_h,
            background="#CCCCCC",
        )
        board_border.grid(row=0, column=0)
        board_border.grid_propagate(False)
        # Inner white holder
        inner_holder = tk.Frame(
            board_border,
            width=DISPLAY_SIZE[0],
            height=DISPLAY_SIZE[1],
            background="#FFFFFF",
        )
        inner_holder.grid(row=0, column=0, padx=border_px, pady=border_px)
        inner_holder.grid_propagate(False)
        self.draw_canvas = tk.Canvas(
            inner_holder,
            width=DISPLAY_SIZE[0],
            height=DISPLAY_SIZE[1],
            bg="#FFFFFF",
            highlightthickness=0,
        )
        self.draw_canvas.grid(row=0, column=0)
        self._bind_drawing_events()
        # PIL image buffer to save drawing
        self.draw_image = Image.new("L", self.draw_logical_size, color=255)
        self.draw_draw = ImageDraw.Draw(self.draw_image)

        # Right-side preview removed; single-predict mode only shows the number

        self.prediction_var = tk.StringVar(value="Draw a number between 1-9")
        self.prediction_label = ttk.Label(container, textvariable=self.prediction_var, font=("Segoe UI", 14))
        self.prediction_label.grid(row=1, column=0, pady=(8, 12))

        # Training progress
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

        # Keep a reference to the current PhotoImage to prevent GC
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Prediction state
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

        self.image_paths = self._collect_digit_images(DIGITS_DIR, FILENAME_PREFIX)
        if not self.image_paths:
            messagebox.showinfo("No images found", f"No images like '{FILENAME_PREFIX}#.png' found in '{DIGITS_DIR}'.")
            return

        try:
            if self.model is None:
                self.model = tf.keras.models.load_model(MODEL_PATH)
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
        # Use the most recently saved image
        image_path = self.image_paths[-1]
        try:
            batch = _model.preprocess_digit_image(image_path)
            assert self.model is not None
            preds = self.model.predict(batch, verbose=0)
            predicted_digit = int(np.argmax(preds))
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
        cell = (max(0, min(self.draw_logical_size[0] - 1, event.x // self.draw_scale)),
                max(0, min(self.draw_logical_size[1] - 1, event.y // self.draw_scale)))
        self._last_cell = cell
        self._paint_cell(cell[0], cell[1])

    def _on_draw_move(self, event) -> None:
        if self._last_cell is None:
            self._on_draw_start(event)
            return
        cell = (max(0, min(self.draw_logical_size[0] - 1, event.x // self.draw_scale)),
                max(0, min(self.draw_logical_size[1] - 1, event.y // self.draw_scale)))
        self._paint_line_cells(self._last_cell, cell)
        self._last_cell = cell

    def _on_draw_end(self, event) -> None:
        self._last_cell = None
        # Auto-save the drawing when the user releases the mouse
        self._auto_save_drawing()
        # Auto-start prediction if not busy
        if not self.is_training and not self.is_running:
            self.root.after(0, self.start_predicting)

    def _paint_cell(self, cx: int, cy: int) -> None:
        x0, y0 = cx * self.draw_scale, cy * self.draw_scale
        x1, y1 = x0 + self.draw_scale, y0 + self.draw_scale
        self.draw_canvas.create_rectangle(x0, y0, x1, y1, outline=self.draw_color, fill=self.draw_color)
        # Update logical image
        self.draw_image.putpixel((cx, cy), 0)

    def _paint_line_cells(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            self._paint_cell(x0, y0)
            return
        for i in range(steps + 1):
            t = i / steps
            cx = int(round(x0 + dx * t))
            cy = int(round(y0 + dy * t))
            if 0 <= cx < self.draw_logical_size[0] and 0 <= cy < self.draw_logical_size[1]:
                self._paint_cell(cx, cy)

    def save_drawing_to_digits(self) -> None:
        # Save the drawing as 28x28 grayscale PNG named digitN.png in digits/
        try:
            os.makedirs(DIGITS_DIR, exist_ok=True)
            # Determine next digit index
            existing = self._collect_digit_images(DIGITS_DIR, FILENAME_PREFIX)
            used_numbers = set()
            pattern = re.compile(rf"^{re.escape(FILENAME_PREFIX)}(\d+)\.png$", re.IGNORECASE)
            for path in existing:
                name = os.path.basename(path)
                m = pattern.match(name)
                if m:
                    used_numbers.add(int(m.group(1)))
            next_idx = (max(used_numbers) + 1) if used_numbers else 1
            save_path = os.path.join(DIGITS_DIR, f"{FILENAME_PREFIX}{next_idx}.png")
            # Save exact 28x28 logical image
            self.draw_image.save(save_path)
            self.temp_drawn_paths.append(save_path)
        except Exception as exc:
            self.prediction_var.set(f"Save error: {exc}")

    def _auto_save_drawing(self) -> None:
        # Wrapper to save without changing status text unless error
        try:
            prev_text = self.prediction_var.get()
            self.save_drawing_to_digits()
            # Restore previous text if it was prediction-related
            if prev_text.startswith("Prediction:"):
                self.prediction_var.set(prev_text)
        except Exception as exc:
            self.prediction_var.set(f"Auto-save error: {exc}")

    def on_close(self) -> None:
        # Cleanup temporary drawn images
        for path in self.temp_drawn_paths:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass
        self.root.destroy()

    def clear_drawing_and_temp(self) -> None:
        # Delete the most recently saved temporary drawing and clear canvas
        try:
            if self.temp_drawn_paths:
                last_path = self.temp_drawn_paths.pop()
                try:
                    if os.path.isfile(last_path):
                        os.remove(last_path)
                except Exception:
                    pass
            # Clear canvas display
            self.draw_canvas.delete("all")
            # Reset logical image buffer
            self.draw_image = Image.new("L", self.draw_logical_size, color=255)
            self.draw_draw = ImageDraw.Draw(self.draw_image)
            self._last_cell = None
            self.prediction_var.set("Cleared.")
        except Exception as exc:
            self.prediction_var.set(f"Clear error: {exc}")

    def train_model_only(self) -> None:
        if self.is_running or self.is_training:
            return
        # Disable actions during training
        self.is_training = True
        self.prediction_var.set("Training model... 0%")
        self.train_progress.set(0.0)
        self.train_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._train_model_task, daemon=True)
        thread.start()

    def _train_model_task(self) -> None:
        try:
            x_train, y_train, x_test, y_test = _model.load_mnist_data()
            x_train_norm, x_test_norm = _model.normalize_images(x_train, x_test)
            model = _model.build_classification_model(input_shape=(28, 28), hidden_units=128, num_classes=10)

            # Keras callback to update progress safely via the Tk thread
            class TkProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, app: "DigitPredictorApp") -> None:
                    super().__init__()
                    self.app = app
                    self.current_epoch = 0
                    self.total_epochs = 1
                    self.steps_per_epoch: int | None = None

                def on_train_begin(self, logs=None):
                    self.total_epochs = int(self.params.get("epochs", 1))
                    # Try to get steps per epoch if available
                    self.steps_per_epoch = self.params.get("steps", None)
                    self._update_percent(0.0)

                def on_epoch_begin(self, epoch, logs=None):
                    self.current_epoch = epoch

                def on_batch_end(self, batch, logs=None):
                    # Smooth progress if steps available
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

            callbacks = [TkProgressCallback(self)]

            # Train with callbacks (quiet output)
            model.fit(x_train_norm, y_train, epochs=3, callbacks=callbacks, verbose=0)
            # Ensure models directory exists
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            _model.save_model_to_disk(model, MODEL_PATH)

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

    @staticmethod
    def _collect_digit_images(directory: str, prefix: str) -> List[str]:
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
        # Sort by numeric suffix (digit1.png, digit2.png, ...)
        candidates.sort(key=lambda x: x[0])
        return [path for _, path in candidates]


def main() -> None:
    root = tk.Tk()
    root.geometry("500x500")
    # Use ttk themes if available
    try:
        style = ttk.Style()
        # Prefer 'vista' on Windows, fallback to default
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except Exception:
        pass
    app = DigitPredictorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()


