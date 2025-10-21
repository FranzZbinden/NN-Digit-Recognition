# GUI helper functions for creating and manipulating Tkinter canvas elements.

import tkinter as tk
from tkinter import ttk

def setup_window_theme(root: tk.Tk) -> None:

    try:
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except Exception:
        pass


def create_drawing_canvas(parent: tk.Widget, display_size: tuple[int, int]) -> tuple[tk.Canvas, tk.Frame]:
    # Create a drawing canvas with border frame.
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


def paint_canvas_cell(canvas: tk.Canvas, cx: int, cy: int, scale: int, color: str = "#000000") -> None:
    # Paint a single cell on the canvas.
    x0, y0 = cx * scale, cy * scale
    x1, y1 = x0 + scale, y0 + scale
    canvas.create_rectangle(x0, y0, x1, y1, outline=color, fill=color)


def compute_line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    # Calculate cells between two points using linear interpolation.
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [(x0, y0)]
    cells: list[tuple[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        cx = int(round(x0 + dx * t))
        cy = int(round(y0 + dy * t))
        cells.append((cx, cy))
    return cells


def paint_canvas_line(canvas: tk.Canvas, start: tuple[int, int], end: tuple[int, int], 
                     scale: int, logical_size: tuple[int, int], color: str = "#000000") -> None:
    # Paint a line between two points on the canvas.
    for cx, cy in compute_line_cells(start, end):
        if 0 <= cx < logical_size[0] and 0 <= cy < logical_size[1]:
            paint_canvas_cell(canvas, cx, cy, scale, color)


def screen_to_logical_coords(event_x: int, event_y: int, scale: int, logical_size: tuple[int, int]) -> tuple[int, int]:
    # Convert screen coordinates to logical grid coordinates.
    cx = max(0, min(logical_size[0] - 1, event_x // scale))
    cy = max(0, min(logical_size[1] - 1, event_y // scale))
    return cx, cy

