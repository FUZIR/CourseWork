"""
Точка входу застосунку.
"""

import tkinter as tk
from tkinter import ttk
from gui.app import SensorPlacementApp


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use('clam')
    except Exception:
        pass
    SensorPlacementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
