# Image management logic for handwritten digit recognition.
# Handles image creation, saving, and file operations.

import os
import re
from typing import List
from PIL import Image

DIGITS_DIR = "digits"
FILENAME_PREFIX = "digit"


def create_drawing_image(logical_size: tuple[int, int]) -> Image.Image:
    # Create a new PIL image for the drawing canvas.
    image = Image.new("L", logical_size, color=255)
    return image


def save_image_to_digits_dir(image: Image.Image, digits_dir: str, filename_prefix: str) -> str:
    # Save image to digits directory with the next available filename.
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


def collect_digit_images(directory: str, prefix: str) -> List[str]:
    # Collect and sort digit image files from a directory.
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


def cleanup_temporary_files(file_paths: List[str]) -> None:
    # Remove temporary files safely.
    for path in file_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass

