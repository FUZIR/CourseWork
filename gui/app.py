"""
Головний GUI-застосунок (tkinter).
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import time
import threading

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from core.utils import generate_problem
from core.io import save_problem, load_problem
from algorithms import greedy_algorithm, tabu_search
from gui.visualization import (
    draw_map, draw_convergence, draw_matrices, draw_comparison_bars)
from gui.dialogs import MatrixEditDialog
from gui import experiments as exp_module


class SensorPlacementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Розміщення датчиків моніторингу довкілля")
        self.root.geometry("1500x920")
        self.root.resizable(True, True)

        self.U = None
        self.S = None

        self._build_ui()

    # ----------------------------------------------------------------
    # Побудова інтерфейсу
    # ----------------------------------------------------------------

    def _build_ui(self):
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(pw, width=370)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        self._build_iz_tab(nb)
        self._build_tabu_tab(nb)
        self._build_exp_tab(nb)

        log_frame = ttk.LabelFrame(parent, text="Лог виконання")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=10, width=40, font=('Courier', 9), wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _build_iz_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="ІЗ")

        fields = [
            ("Рядків (m):",          "m",          "6"),
            ("Стовпців (n):",        "n",          "8"),
            ("Вартість A (α):",      "alpha",      "100"),
            ("Вартість B (β):",      "beta",       "200"),
            ("Бюджет (L):",          "budget",     "500"),
            ("Макс. датч. B (h):",   "h_max",      "1"),
            ("Ліміт покриття (λ):",  "lambda_lim", "1"),
            ("Щільність S (d):",     "density",    "0.75"),
            ("Макс. вага U (u_max):","u_max",      "50"),
        ]
        self.params = {}
        for row, (label, key, default) in enumerate(fields):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky='w', padx=6, pady=2)
            var = tk.StringVar(value=default)
            ttk.Entry(tab, textvariable=var, width=12).grid(
                row=row, column=1, padx=6, pady=2)
            self.params[key] = var

        r = len(fields)
        ttk.Separator(tab, orient='horizontal').grid(
            row=r, column=0, columnspan=2, sticky='ew', pady=4)

        btn = ttk.Frame(tab)
        btn.grid(row=r + 1, column=0, columnspan=2, sticky='ew', padx=6)

        ttk.Button(btn, text="Згенерувати задачу",
                   command=self.generate_problem).pack(fill=tk.X, pady=2)

        row2 = ttk.Frame(btn)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="Редагувати U",
                   command=self.edit_matrix_u).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(row2, text="Редагувати S",
                   command=self.edit_matrix_s).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        row3 = ttk.Frame(btn)
        row3.pack(fill=tk.X, pady=2)
        ttk.Button(row3, text="Зберегти задачу",
                   command=self.save_to_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(row3, text="Завантажити задачу",
                   command=self.load_from_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        ttk.Separator(btn, orient='horizontal').pack(fill=tk.X, pady=4)
        ttk.Button(btn, text="▶  Жадібний алгоритм",
                   command=lambda: self.run_algorithm('greedy')).pack(fill=tk.X, pady=2)

    def _build_tabu_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Табу-пошук")

        tabu_fields = [
            ("Розмір списку табу:", "tabu_size", "10"),
        ]
        self.tabu_params = {}
        for row, (label, key, default) in enumerate(tabu_fields):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky='w', padx=6, pady=4)
            var = tk.StringVar(value=default)
            ttk.Entry(tab, textvariable=var, width=10).grid(
                row=row, column=1, padx=6, pady=4)
            self.tabu_params[key] = var

        btn = ttk.Frame(tab)
        btn.grid(row=len(tabu_fields) + 1, column=0, columnspan=2, sticky='ew', padx=6, pady=4)
        ttk.Button(btn, text="▶  Табу-пошук",
                   command=lambda: self.run_algorithm('tabu')).pack(fill=tk.X, pady=2)
        ttk.Button(btn, text="▶▶  Порівняти обидва алгоритми",
                   command=self.compare_algorithms).pack(fill=tk.X, pady=2)

    def _build_exp_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Експерименти")

        # Scrollable container
        canvas = tk.Canvas(tab, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(
            scrollregion=canvas.bbox('all')))
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(
            -1 if e.delta > 0 else 1, 'units'))

        self.exp_ranges = {}

        # ---- Group: Розмірність задачі ----
        self._exp_group(inner, "Розмірність задачі (dim×dim)", [
            ("Від:",        'size_from',  '4'),
            ("До:",         'size_to',    '12'),
            ("Крок:",       'size_step',  '2'),
            ("ІЗ на розмірність:", 'size_tasks', '5'),
        ], self.experiment_size)

        # ---- Group: Вплив λ ----
        self._exp_group(inner, "Вплив параметра λ", [
            ("λ від:",  'lam_from',  '1'),
            ("λ до:",   'lam_to',    '5'),
            ("Крок:",   'lam_step',  '1'),
            ("К-сть ІЗ:", 'lam_tasks', '5'),
        ], self.experiment_lambda)

        # ---- Group: Коефіцієнт K (MaxIter = K·m·n) ----
        self._exp_group(inner, "Коефіцієнт K  (MaxIter = K·m·n)", [
            ("K від:",      'iter_from',     '1'),
            ("K до:",       'iter_to',       '10'),
            ("Крок K:",     'iter_step',     '1'),
            ("dim від:",    'iter_dim_from', '4'),
            ("dim до:",     'iter_dim_to',   '10'),
            ("Крок dim:",   'iter_dim_step', '2'),
            ("К-сть ІЗ:",   'iter_tasks',    '3'),
        ], self.experiment_max_iter)

        # ---- Group: Розмір списку табу ----
        self._exp_group(inner, "Розмір списку табу (Tsize)", [
            ("Tsize від:", 'tsize_from',  '3'),
            ("Tsize до:",  'tsize_to',    '20'),
            ("Крок:",      'tsize_step',  '3'),
            ("К-сть ІЗ:",  'tsize_tasks', '5'),
        ], self.experiment_tabu_size)

        # ---- Group: Порівняння алгоритмів ----
        self._exp_group(inner, "Порівняння алгоритмів", [
            ("К-сть ІЗ:", 'cmp_tasks', '10'),
        ], self.experiment_comparison)

    def _exp_group(self, parent, title, fields, cmd):
        """Створює LabelFrame-групу з полями від/до/крок та кнопкою запуску."""
        lf = ttk.LabelFrame(parent, text=title)
        lf.pack(fill=tk.X, padx=6, pady=4)

        for row, (label, key, default) in enumerate(fields):
            ttk.Label(lf, text=label, width=18, anchor='w').grid(
                row=row, column=0, sticky='w', padx=6, pady=2)
            var = tk.StringVar(value=default)
            ttk.Entry(lf, textvariable=var, width=8).grid(
                row=row, column=1, padx=4, pady=2)
            self.exp_ranges[key] = var

        ttk.Button(lf, text="▶ Запустити", command=cmd).grid(
            row=len(fields), column=0, columnspan=2, sticky='ew', padx=6, pady=4)

    def _build_right_panel(self, parent):
        self.viz_nb = ttk.Notebook(parent)
        self.viz_nb.pack(fill=tk.BOTH, expand=True)

        for tab_title, fig_attr, canvas_attr in [
            ("  Карта розміщення  ",      'fig_map',      'canvas_map'),
            ("  Збіжність  ",             'fig_conv',     'canvas_conv'),
            ("  Графіки експериментів  ", 'fig_exp',      'canvas_exp'),
            ("  Матриці U та S  ",        'fig_matrices', 'canvas_matrices'),
        ]:
            frame = ttk.Frame(self.viz_nb)
            self.viz_nb.add(frame, text=tab_title)
            fig = plt.Figure(figsize=(9, 6), tight_layout=True)
            setattr(self, fig_attr, fig)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            setattr(self, canvas_attr, canvas)

        self.ax_conv = self.fig_conv.add_subplot(111)

    # ----------------------------------------------------------------
    # Утиліти
    # ----------------------------------------------------------------

    def log_msg(self, msg):
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self.log_msg, msg)
            return
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def _run_async(self, worker, on_done):
        """Запускає worker() у фоновому потоці; після завершення викликає on_done(result) у головному потоці."""
        def _thread():
            result = worker()
            self.root.after(0, on_done, result)
        threading.Thread(target=_thread, daemon=True).start()

    def get_params(self):
        try:
            m          = int(self.params['m'].get())
            n          = int(self.params['n'].get())
            alpha      = float(self.params['alpha'].get())
            beta       = float(self.params['beta'].get())
            L          = float(self.params['budget'].get())
            h          = int(self.params['h_max'].get())
            lambda_lim = int(self.params['lambda_lim'].get())
            if m <= 0 or n <= 0:
                raise ValueError("m та n мають бути > 0")
            if alpha <= 0 or beta <= alpha:
                raise ValueError("Потрібно β > α > 0")
            if L < alpha:
                raise ValueError("Бюджет L має бути не менше α")
            return m, n, alpha, beta, L, h, lambda_lim
        except ValueError as e:
            messagebox.showerror("Помилка параметрів", str(e))
            return None

    def _get_density(self):
        try:
            return float(self.params['density'].get())
        except ValueError:
            return 0.75

    def _get_u_max(self):
        try:
            return int(self.params['u_max'].get())
        except ValueError:
            return 50


    def _ensure_matrices(self, m, n, alpha, beta):
        if self.U is None or self.U.shape != (m, n):
            d = self._get_density()
            u_max = self._get_u_max()
            self.U, self.S, _, _, _ = generate_problem(m, n, alpha, beta, d=d, u_max=u_max)
            self._redraw_matrices()

    def _irange(self, key_from, key_to, key_step):
        """Повертає range із значень полів from/to/step."""
        return range(
            int(self.exp_ranges[key_from].get()),
            int(self.exp_ranges[key_to].get()) + 1,
            max(1, int(self.exp_ranges[key_step].get())),
        )

    def _itasks(self, key):
        return max(1, int(self.exp_ranges[key].get()))

    # ----------------------------------------------------------------
    # Дії кнопок ІЗ
    # ----------------------------------------------------------------

    def generate_problem(self):
        p = self.get_params()
        if p is None:
            return
        m, n, alpha, beta, *_ = p
        d = self._get_density()
        u_max = self._get_u_max()
        self.U, self.S, _, _, _ = generate_problem(m, n, alpha, beta, d=d, u_max=u_max)
        self._redraw_matrices()
        self._redraw_map([], [], None, "Нова задача (без датчиків)")
        self.log_msg(f"\nЗгенеровано задачу {m}×{n}, d={d}, u_max={u_max}")

    def edit_matrix_u(self):
        if self.U is None:
            messagebox.showinfo("Редагування", "Спочатку згенеруйте задачу.")
            return
        dlg = MatrixEditDialog(self.root, self.U, mode='U')
        if dlg.result is not None:
            self.U = dlg.result
            self._redraw_matrices()
            self._redraw_map([], [], None, "Оновлено матрицю U")
            self.log_msg("Матриця U оновлена вручну.")

    def edit_matrix_s(self):
        if self.S is None:
            messagebox.showinfo("Редагування", "Спочатку згенеруйте задачу.")
            return
        dlg = MatrixEditDialog(self.root, self.S.astype(float), mode='S')
        if dlg.result is not None:
            self.S = dlg.result.astype(int)
            self._redraw_matrices()
            self._redraw_map([], [], None, "Оновлено матрицю S")
            self.log_msg("Матриця S оновлена вручну.")

    def save_to_file(self):
        if self.U is None:
            messagebox.showinfo("Збереження", "Немає задачі для збереження.")
            return
        p = self.get_params()
        if p is None:
            return
        _, _, alpha, beta, L, h, lambda_lim = p
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файли", "*.json"), ("Усі файли", "*.*")],
            title="Зберегти задачу",
        )
        if not path:
            return
        params = {'alpha': alpha, 'beta': beta, 'L': L, 'h': h, 'lambda_lim': lambda_lim}
        try:
            save_problem(path, self.U, self.S, params)
            self.log_msg(f"Задачу збережено: {path}")
        except Exception as exc:
            messagebox.showerror("Помилка збереження", str(exc))

    def load_from_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON файли", "*.json"), ("Усі файли", "*.*")],
            title="Завантажити задачу",
        )
        if not path:
            return
        try:
            U, S, params = load_problem(path)
        except Exception as exc:
            messagebox.showerror("Помилка завантаження", str(exc))
            return

        self.U = U
        self.S = S
        m, n = U.shape

        # Update UI fields
        self.params['m'].set(str(m))
        self.params['n'].set(str(n))
        for key in ('alpha', 'beta', 'L', 'h', 'lambda_lim'):
            if key in params:
                ui_key = 'budget' if key == 'L' else ('h_max' if key == 'h' else key)
                val = params[key]
                self.params[ui_key].set(
                    str(int(val)) if float(val) == int(float(val)) else str(val))

        self._redraw_matrices()
        self._redraw_map([], [], None, f"Завантажено: {path.split('/')[-1]}")
        self.log_msg(f"Задачу завантажено: {path}")
        self.log_msg(f"  Розмір: {m}×{n}")

    # ----------------------------------------------------------------
    # Запуск алгоритмів
    # ----------------------------------------------------------------

    def run_algorithm(self, algo):
        p = self.get_params()
        if p is None:
            return
        m, n, alpha, beta, L, h, lambda_lim = p
        self._ensure_matrices(m, n, alpha, beta)
        self.log_msg("\n" + "=" * 40)

        if algo == 'greedy':
            t0 = time.time()
            A, B, F, K, history = greedy_algorithm(
                self.U, self.S, m, n, alpha, beta, L, h, lambda_lim)
            elapsed = time.time() - t0
            self._log_result("ЖАДІБНИЙ АЛГОРИТМ", A, B, F, alpha, beta, L, elapsed,
                             extra=f"  Ітерацій: {len(history)}")
            self._redraw_map(A, B, K,
                             f"Жадібний | F={F} | Витрати={len(A)*alpha+len(B)*beta:.0f}/{L:.0f}")
            f_vals = [entry['F'] for entry in history]
            self._redraw_convergence(f_vals, f_vals, "Жадібний алгоритм")

        elif algo == 'tabu':
            max_iter  = m * n
            tabu_size = int(self.tabu_params['tabu_size'].get())
            t0 = time.time()
            A, B, F, K, history = tabu_search(
                self.U, self.S, m, n, alpha, beta, L, h, lambda_lim,
                max_iter=max_iter, tabu_size=tabu_size)
            elapsed = time.time() - t0
            self._log_result("ТАБУ-ПОШУК", A, B, F, alpha, beta, L, elapsed,
                             extra=f"  Ітерацій: {len(history)}")
            self._redraw_map(A, B, K,
                             f"Табу-пошук | F={F} | Витрати={len(A)*alpha+len(B)*beta:.0f}/{L:.0f}")
            curr_f = [entry['F'] for entry in history]
            best_f = [entry['best_F'] for entry in history]
            self._redraw_convergence(curr_f, best_f, "Табу-пошук")

    def compare_algorithms(self):
        p = self.get_params()
        if p is None:
            return
        m, n, alpha, beta, L, h, lambda_lim = p
        self._ensure_matrices(m, n, alpha, beta)
        max_iter  = m * n
        tabu_size = int(self.tabu_params['tabu_size'].get())
        U, S = self.U, self.S

        def worker():
            t0 = time.time()
            _, _, F_g, _, _ = greedy_algorithm(U, S, m, n, alpha, beta, L, h, lambda_lim)
            t_g = time.time() - t0
            t0 = time.time()
            _, _, F_t, _, _ = tabu_search(U, S, m, n, alpha, beta, L, h, lambda_lim,
                                          max_iter=max_iter, tabu_size=tabu_size)
            t_tabu = time.time() - t0
            return F_g, F_t, t_g, t_tabu

        def on_done(result):
            F_g, F_t, t_g, t_tabu = result
            improve = (F_t - F_g) / F_g * 100 if F_g > 0 else 0.0
            self.log_msg("\n" + "=" * 40)
            self.log_msg("ПОРІВНЯННЯ АЛГОРИТМІВ")
            self.log_msg(f"  Жадібний:   F={F_g}, час={t_g:.4f}с")
            self.log_msg(f"  Табу-пошук: F={F_t}, час={t_tabu:.4f}с")
            self.log_msg(f"  Покращення: {improve:+.1f}%")
            draw_comparison_bars(self.fig_exp, [F_g, F_t], [t_g, t_tabu])
            self.fig_exp.suptitle(
                f"Порівняння | покращення табу: {improve:+.1f}%",
                fontsize=11, fontweight='bold')
            self.fig_exp.tight_layout()
            self.canvas_exp.draw()
            self.viz_nb.select(2)

        self._run_async(worker, on_done)

    # ----------------------------------------------------------------
    # Спільний контекст для експериментів
    # ----------------------------------------------------------------

    def _exp_context(self):
        """Повертає (m, n, alpha, beta, L, h, lambda_lim, max_iter, tabu_size, d) або None."""
        p = self.get_params()
        if p is None:
            return None
        m, n, alpha, beta, L, h, lambda_lim = p
        max_iter  = m * n
        tabu_size = int(self.tabu_params['tabu_size'].get())
        d = self._get_density()
        return m, n, alpha, beta, L, h, lambda_lim, max_iter, tabu_size, d

    # ----------------------------------------------------------------
    # Експерименти
    # ----------------------------------------------------------------

    def experiment_size(self):
        ctx = self._exp_context()
        if ctx is None:
            return
        _, _, alpha, beta, _, h, lambda_lim, max_iter, tabu_size, d = ctx
        dim_from  = int(self.exp_ranges['size_from'].get())
        dim_to    = int(self.exp_ranges['size_to'].get())
        dim_step  = max(1, int(self.exp_ranges['size_step'].get()))
        tasks     = self._itasks('size_tasks')

        def worker():
            return exp_module.run_size_experiment(
                alpha, beta, h, lambda_lim, tasks, d,
                max_iter, tabu_size, dim_from, dim_to, dim_step, self.log_msg)

        def on_done(result):
            labels, gf, tf, gt, tt = result
            if not labels:
                return
            x = np.arange(len(labels))
            self.fig_exp.clear()
            ax1 = self.fig_exp.add_subplot(121)
            ax2 = self.fig_exp.add_subplot(122)
            w = 0.35
            ax1.bar(x - w/2, gf, w, label='Жадібний',   color='#2196F3')
            ax1.bar(x + w/2, tf, w, label='Табу-пошук', color='#FF5722')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=30)
            ax1.set_title("Цільова функція F")
            ax1.set_ylabel("F (середнє)")
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax2.plot(x, gt, 'b-o', label='Жадібний')
            ax2.plot(x, tt, 'r-s', label='Табу-пошук')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=30)
            ax2.set_title("Час виконання")
            ax2.set_ylabel("час, с")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            self.fig_exp.suptitle("Вплив розмірності задачі", fontsize=12, fontweight='bold')
            self.fig_exp.tight_layout()
            self.canvas_exp.draw()
            self.viz_nb.select(2)

        self._run_async(worker, on_done)

    def experiment_lambda(self):
        ctx = self._exp_context()
        if ctx is None:
            return
        m, n, alpha, beta, L, h, _, max_iter, tabu_size, d = ctx
        lam_from = int(self.exp_ranges['lam_from'].get())
        lam_to   = int(self.exp_ranges['lam_to'].get())
        lam_step = max(1, int(self.exp_ranges['lam_step'].get()))
        runs     = self._itasks('lam_tasks')

        def worker():
            return exp_module.run_lambda_experiment(
                m, n, alpha, beta, L, h, runs, d,
                max_iter, tabu_size, lam_from, lam_to, lam_step, self.log_msg)

        def on_done(result):
            lambdas, gf, tf = result
            if not lambdas:
                return
            self.fig_exp.clear()
            ax = self.fig_exp.add_subplot(111)
            ax.plot(lambdas, gf, 'b-o', label='Жадібний',   linewidth=2, markersize=8)
            ax.plot(lambdas, tf, 'r-s', label='Табу-пошук', linewidth=2, markersize=8)
            ax.set_xlabel("Ліміт перекриття λ")
            ax.set_ylabel("F (середнє)")
            ax.set_title("Вплив параметра λ", fontsize=12, fontweight='bold')
            ax.set_xticks(lambdas)
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.fig_exp.tight_layout()
            self.canvas_exp.draw()
            self.viz_nb.select(2)

        self._run_async(worker, on_done)

    def experiment_max_iter(self):
        ctx = self._exp_context()
        if ctx is None:
            return
        _, _, alpha, beta, _, h, lambda_lim, _, tabu_size, d = ctx
        k_from   = int(self.exp_ranges['iter_from'].get())
        k_to     = int(self.exp_ranges['iter_to'].get())
        k_step   = max(1, int(self.exp_ranges['iter_step'].get()))
        dim_from = int(self.exp_ranges['iter_dim_from'].get())
        dim_to   = int(self.exp_ranges['iter_dim_to'].get())
        dim_step = max(1, int(self.exp_ranges['iter_dim_step'].get()))
        runs     = self._itasks('iter_tasks')

        def worker():
            return exp_module.run_max_iter_experiment(
                alpha, beta, h, lambda_lim,
                tabu_size, runs, d,
                k_from, k_to, k_step,
                dim_from, dim_to, dim_step, self.log_msg)

        def on_done(result):
            dims, ks, all_fs, all_ts = result
            if not ks or not dims:
                return
            colors = plt.cm.tab10(np.linspace(0, 1, len(dims)))
            self.fig_exp.clear()
            ax1 = self.fig_exp.add_subplot(121)
            ax2 = self.fig_exp.add_subplot(122)
            for i, dim in enumerate(dims):
                lbl = f"{dim}×{dim}"
                ax1.plot(ks, all_fs[dim], '-s', color=colors[i], label=lbl,
                         linewidth=2, markersize=7)
                ax2.plot(ks, all_ts[dim], '-^', color=colors[i], label=lbl,
                         linewidth=2, markersize=7)
            ax1.set_xlabel("K  (MaxIter = K·dim²)")
            ax1.set_ylabel("F (середнє)")
            ax1.set_title("Якість від K та розмірності")
            ax1.set_xticks(ks)
            ax1.legend(title="dim")
            ax1.grid(True, alpha=0.3)
            ax2.set_xlabel("K  (MaxIter = K·dim²)")
            ax2.set_ylabel("час, с")
            ax2.set_title("Час від K та розмірності")
            ax2.set_xticks(ks)
            ax2.legend(title="dim")
            ax2.grid(True, alpha=0.3)
            self.fig_exp.suptitle(
                "Вплив K та розмірності (MaxIter = K·dim²)",
                fontsize=12, fontweight='bold')
            self.fig_exp.tight_layout()
            self.canvas_exp.draw()
            self.viz_nb.select(2)

        self._run_async(worker, on_done)

    def experiment_tabu_size(self):
        ctx = self._exp_context()
        if ctx is None:
            return
        m, n, alpha, beta, L, h, lambda_lim, max_iter, _, d = ctx
        tsize_from = int(self.exp_ranges['tsize_from'].get())
        tsize_to   = int(self.exp_ranges['tsize_to'].get())
        tsize_step = max(1, int(self.exp_ranges['tsize_step'].get()))
        runs       = self._itasks('tsize_tasks')

        def worker():
            return exp_module.run_tabu_size_experiment(
                m, n, alpha, beta, L, h, lambda_lim, runs, d,
                max_iter, tsize_from, tsize_to, tsize_step, self.log_msg)

        def on_done(result):
            sizes, fs, ts = result
            if not sizes:
                return
            self.fig_exp.clear()
            ax1 = self.fig_exp.add_subplot(121)
            ax2 = self.fig_exp.add_subplot(122)
            ax1.plot(sizes, fs, 'r-s', linewidth=2, markersize=8)
            ax1.set_xlabel("Розмір списку табу")
            ax1.set_ylabel("F (середнє)")
            ax1.set_title("Якість від Tsize")
            ax1.grid(True, alpha=0.3)
            ax2.plot(sizes, ts, 'g-^', linewidth=2, markersize=8)
            ax2.set_xlabel("Розмір списку табу")
            ax2.set_ylabel("час, с")
            ax2.set_title("Час від Tsize")
            ax2.grid(True, alpha=0.3)
            self.fig_exp.suptitle("Вплив розміру списку табу", fontsize=12, fontweight='bold')
            self.fig_exp.tight_layout()
            self.canvas_exp.draw()
            self.viz_nb.select(2)

        self._run_async(worker, on_done)

    def experiment_comparison(self):
        ctx = self._exp_context()
        if ctx is None:
            return
        m, n, alpha, beta, L, h, lambda_lim, max_iter, tabu_size, d = ctx
        runs = self._itasks('cmp_tasks')

        def worker():
            return exp_module.run_comparison_experiment(
                m, n, alpha, beta, L, h, lambda_lim,
                max_iter, tabu_size, runs, d, self.log_msg)

        def on_done(result):
            gf, tf, gt, tt = result
            task_ids = list(range(1, runs + 1))
            self.fig_exp.clear()
            ax1 = self.fig_exp.add_subplot(121)
            ax2 = self.fig_exp.add_subplot(122)
            ax1.plot(task_ids, gf, 'b-o', label='Жадібний',   linewidth=1.5, markersize=7)
            ax1.plot(task_ids, tf, 'r-s', label='Табу-пошук', linewidth=1.5, markersize=7)
            ax1.set_xlabel("Номер ІЗ")
            ax1.set_ylabel("F")
            ax1.set_title("Цільова функція F по ІЗ")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            labels = ['Жадібний', 'Табу-пошук']
            colors = ['#2196F3', '#FF5722']
            bars = ax2.bar(labels, [np.mean(gt), np.mean(tt)], color=colors,
                           width=0.5, edgecolor='black')
            ax2.set_title("Середній час виконання")
            ax2.set_ylabel("час, с")
            for bar, val in zip(bars, [np.mean(gt), np.mean(tt)]):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.4f}", ha='center', va='bottom', fontsize=10)
            improve = ((np.mean(tf) - np.mean(gf)) / np.mean(gf) * 100
                       if np.mean(gf) > 0 else 0.0)
            self.fig_exp.suptitle(
                f"Порівняння алгоритмів | покращення табу: {improve:+.1f}%",
                fontsize=11, fontweight='bold')
            self.fig_exp.tight_layout()
            self.canvas_exp.draw()
            self.viz_nb.select(2)

        self._run_async(worker, on_done)

    # ----------------------------------------------------------------
    # Внутрішні методи відображення
    # ----------------------------------------------------------------

    def _log_result(self, header, A, B, F, alpha, beta, L, elapsed, extra=""):
        cost = len(A) * alpha + len(B) * beta
        self.log_msg(header)
        self.log_msg(f"  Час:     {elapsed:.4f} с")
        self.log_msg(f"  Датчики A ({len(A)}): {[(i+1, j+1) for (i,j) in A]}")
        self.log_msg(f"  Датчики B ({len(B)}): {[(i+1, j+1) for (i,j) in B]}")
        self.log_msg(f"  F = {F}")
        self.log_msg(f"  Витрати: {cost:.0f} / {L:.0f}")
        if extra:
            self.log_msg(extra)

    def _redraw_map(self, A, B, K, title):
        if self.U is None:
            return
        draw_map(self.fig_map, self.U, self.S, A, B, K, title)
        self.canvas_map.draw()
        self.viz_nb.select(0)

    def _redraw_convergence(self, curr_f, best_f, title):
        draw_convergence(self.fig_conv, self.ax_conv, curr_f, best_f, title)
        self.canvas_conv.draw()
        self.viz_nb.select(1)

    def _redraw_matrices(self):
        if self.U is None:
            return
        draw_matrices(self.fig_matrices, self.U, self.S)
        self.canvas_matrices.draw()
