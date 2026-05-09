"""
Діалогові вікна для редагування матриць U та S.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


class MatrixEditDialog(tk.Toplevel):
    """
    Модальне вікно для редагування матриці.

    mode='U' — числові ваги (цілі від 1 до 50)
    mode='S' — матриця доступності (0 або 1)
    """

    def __init__(self, parent, matrix, mode='U'):
        super().__init__(parent)
        self.result = None
        self.mode = mode
        self.matrix = matrix.copy()
        m, n = matrix.shape

        title = ("Редагування матриці ваг U"
                 if mode == 'U' else "Редагування матриці доступності S")
        self.title(title)
        self.resizable(True, True)
        self.grab_set()
        self.transient(parent)

        hint = ("Введіть цілі числа від 1 до 50"
                if mode == 'U' else "Введіть 0 (заборонено) або 1 (доступно)")
        ttk.Label(self, text=hint, foreground='#555555').pack(pady=4)

        # Scrollable area
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        c = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=c.yview)
        hsb = ttk.Scrollbar(outer, orient='horizontal', command=c.xview)
        c.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        grid_f = ttk.Frame(c)
        c.create_window((0, 0), window=grid_f, anchor='nw')
        grid_f.bind('<Configure>', lambda e: c.configure(scrollregion=c.bbox('all')))

        # Column headers
        ttk.Label(grid_f, text="i\\j", width=4, anchor='center').grid(row=0, column=0)
        for j in range(n):
            ttk.Label(grid_f, text=str(j + 1), width=5, anchor='center').grid(
                row=0, column=j + 1)

        # Entry grid
        self.entries = []
        for i in range(m):
            ttk.Label(grid_f, text=str(i + 1), width=4, anchor='center').grid(
                row=i + 1, column=0)
            row_vars = []
            for j in range(n):
                var = tk.StringVar(value=str(int(matrix[i][j])))
                e = ttk.Entry(grid_f, textvariable=var, width=5, justify='center')
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                row_vars.append(var)
            self.entries.append(row_vars)

        # Buttons
        btn_f = ttk.Frame(self)
        btn_f.pack(pady=6)
        ttk.Button(btn_f, text="OK", command=self._ok, width=10).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btn_f, text="Скасувати", command=self.destroy, width=10).pack(
            side=tk.LEFT, padx=6)

        self.update_idletasks()
        w = min(120 + n * 56, 900)
        h = min(160 + m * 29, 680)
        self.geometry(f"{w}x{h}")
        self.wait_window()

    def _ok(self):
        m, n = self.matrix.shape
        dtype = float if self.mode == 'U' else int
        new_mat = np.zeros((m, n), dtype=dtype)
        try:
            for i in range(m):
                for j in range(n):
                    val = int(self.entries[i][j].get())
                    if self.mode == 'U' and not (1 <= val <= 50):
                        raise ValueError(f"U[{i+1}][{j+1}] має бути від 1 до 50")
                    if self.mode == 'S' and val not in (0, 1):
                        raise ValueError(f"S[{i+1}][{j+1}] має бути 0 або 1")
                    new_mat[i][j] = val
        except ValueError as exc:
            messagebox.showerror("Помилка введення", str(exc), parent=self)
            return

        if self.mode == 'S' and int(np.sum(new_mat)) == 0:
            messagebox.showerror(
                "Помилка", "Матриця S не може бути повністю нульовою", parent=self)
            return

        self.result = new_mat
        self.destroy()
