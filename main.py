# Entry point for handwritten digit recognition application.

import tkinter as tk
from gui_helpers import setup_window_theme
from app.digit_predictor_app import DigitPredictorApp


def main() -> None:
    root = tk.Tk()
    root.geometry("500x500")
    
    setup_window_theme(root)
    
    app = DigitPredictorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
