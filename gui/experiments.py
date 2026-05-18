"""
Функції для проведення обчислювальних експериментів.
"""

import time
import numpy as np

from core.utils import generate_problem
from algorithms import greedy_algorithm, tabu_search


def run_size_experiment(alpha, beta, h, lambda_lim, tasks_per_size, d,
                        _max_iter_unused, tabu_size,
                        dim_from, dim_to, dim_step, log_fn):
    """Дослідження впливу розмірності задачі (квадратні сітки dim×dim)."""
    dims = list(range(dim_from, dim_to + 1, dim_step))
    if not dims:
        log_fn("  Порожній діапазон розмірностей")
        return [], [], [], [], []

    greedy_fs, tabu_fs, greedy_ts, tabu_ts = [], [], [], []

    log_fn("\nЕксперимент: вплив розмірності")
    for dim in dims:
        m_i, n_i = dim, dim
        L_i = alpha * max(3, int(d * m_i * n_i * 0.3))
        gf, tf, gt, tt = [], [], [], []
        for seed in range(tasks_per_size):
            U_i, S_i, _, _, _ = generate_problem(m_i, n_i, alpha, beta, d=d, seed=seed)
            t0 = time.time()
            _, _, F_g, _, _ = greedy_algorithm(
                U_i, S_i, m_i, n_i, alpha, beta, L_i, h, lambda_lim)
            gt.append(time.time() - t0)
            gf.append(F_g)
            t0 = time.time()
            _, _, F_t, _, _ = tabu_search(
                U_i, S_i, m_i, n_i, alpha, beta, L_i, h, lambda_lim,
                max_iter=m_i * n_i, tabu_size=tabu_size)
            tt.append(time.time() - t0)
            tf.append(F_t)
        greedy_fs.append(np.mean(gf))
        tabu_fs.append(np.mean(tf))
        greedy_ts.append(np.mean(gt))
        tabu_ts.append(np.mean(tt))
        log_fn(f"  {m_i}×{n_i}: Жадібний F={greedy_fs[-1]:.1f} ({greedy_ts[-1]:.3f}с), "
               f"Табу F={tabu_fs[-1]:.1f} ({tabu_ts[-1]:.3f}с)")

    labels = [f"{d}×{d}" for d in dims]
    return labels, greedy_fs, tabu_fs, greedy_ts, tabu_ts


def run_lambda_experiment(m, n, alpha, beta, L, h, runs, d,
                          max_iter, tabu_size,
                          lam_from, lam_to, lam_step, log_fn):
    """Дослідження впливу параметра λ."""
    lambdas = list(range(lam_from, lam_to + 1, lam_step))
    if not lambdas:
        log_fn("  Порожній діапазон λ")
        return [], [], []

    greedy_fs, tabu_fs = [], []

    log_fn("\nЕксперимент: вплив λ")
    for lam in lambdas:
        gf, tf = [], []
        for seed in range(runs):
            U_i, S_i, _, _, _ = generate_problem(m, n, alpha, beta, d=d, seed=seed)
            _, _, F_g, _, _ = greedy_algorithm(U_i, S_i, m, n, alpha, beta, L, h, lam)
            _, _, F_t, _, _ = tabu_search(
                U_i, S_i, m, n, alpha, beta, L, h, lam,
                max_iter=max_iter, tabu_size=tabu_size)
            gf.append(F_g)
            tf.append(F_t)
        greedy_fs.append(np.mean(gf))
        tabu_fs.append(np.mean(tf))
        log_fn(f"  λ={lam}: Жадібний={greedy_fs[-1]:.1f}, Табу={tabu_fs[-1]:.1f}")

    return lambdas, greedy_fs, tabu_fs


def run_tabu_size_experiment(m, n, alpha, beta, L, h, lambda_lim,
                              runs, d, max_iter,
                              tsize_from, tsize_to, tsize_step, log_fn):
    """Дослідження впливу розміру списку табу."""
    sizes = list(range(tsize_from, tsize_to + 1, tsize_step))
    if not sizes:
        log_fn("  Порожній діапазон Tsize")
        return [], [], []

    fs, ts = [], []

    log_fn("\nЕксперимент: вплив розміру списку табу")
    for tsize in sizes:
        f_list, t_list = [], []
        for seed in range(runs):
            U_i, S_i, _, _, _ = generate_problem(m, n, alpha, beta, d=d, seed=seed)
            t0 = time.time()
            _, _, F_t, _, _ = tabu_search(
                U_i, S_i, m, n, alpha, beta, L, h, lambda_lim,
                max_iter=max_iter, tabu_size=tsize)
            t_list.append(time.time() - t0)
            f_list.append(F_t)
        fs.append(np.mean(f_list))
        ts.append(np.mean(t_list))
        log_fn(f"  Tsize={tsize}: F={fs[-1]:.1f}, час={ts[-1]:.3f}с")

    return sizes, fs, ts


def run_max_iter_experiment(alpha, beta, h, lambda_lim,
                             tabu_size, runs, d,
                             k_from, k_to, k_step,
                             dim_from, dim_to, dim_step, log_fn):
    """Дослідження впливу K та розмірності (MaxIter = K·dim², m=n=dim)."""
    ks = list(range(k_from, k_to + 1, k_step))
    dims = list(range(dim_from, dim_to + 1, dim_step))

    if not ks:
        log_fn("  Порожній діапазон K")
        return [], [], {}, {}
    if not dims:
        log_fn("  Порожній діапазон розмірностей")
        return [], [], {}, {}

    all_fs = {}
    all_ts = {}

    log_fn("\nЕксперимент: вплив K та розмірності (MaxIter = K·dim²)")
    for dim in dims:
        m_i, n_i = dim, dim
        L_i = alpha * max(3, int(d * m_i * n_i * 0.3))
        fs, ts = [], []
        for k in ks:
            max_it = max(1, k * m_i * n_i)
            f_list, t_list = [], []
            for seed in range(runs):
                U_i, S_i, _, _, _ = generate_problem(m_i, n_i, alpha, beta, d=d, seed=seed)
                t0 = time.time()
                _, _, F_t, _, _ = tabu_search(
                    U_i, S_i, m_i, n_i, alpha, beta, L_i, h, lambda_lim,
                    max_iter=max_it, tabu_size=tabu_size)
                t_list.append(time.time() - t0)
                f_list.append(F_t)
            fs.append(np.mean(f_list))
            ts.append(np.mean(t_list))
            log_fn(f"  {m_i}×{n_i}, K={k} (MaxIter={max_it}): F={fs[-1]:.1f}, час={ts[-1]:.3f}с")
        all_fs[dim] = fs
        all_ts[dim] = ts

    return dims, ks, all_fs, all_ts


def run_comparison_experiment(m, n, alpha, beta, L, h, lambda_lim,
                               max_iter, tabu_size, runs, d, log_fn):
    """
    Порівняння алгоритмів на runs випадкових задачах.

    Повертає (greedy_fs, tabu_fs, greedy_ts, tabu_ts) — списки по runs значень.
    """
    greedy_fs, tabu_fs, greedy_ts, tabu_ts = [], [], [], []

    log_fn("\nЕксперимент: порівняння алгоритмів")
    for seed in range(runs):
        U_i, S_i, _, _, _ = generate_problem(m, n, alpha, beta, d=d, seed=seed)
        t0 = time.time()
        _, _, F_g, _, _ = greedy_algorithm(U_i, S_i, m, n, alpha, beta, L, h, lambda_lim)
        greedy_ts.append(time.time() - t0)
        greedy_fs.append(F_g)
        t0 = time.time()
        _, _, F_t, _, _ = tabu_search(
            U_i, S_i, m, n, alpha, beta, L, h, lambda_lim,
            max_iter=max_iter, tabu_size=tabu_size)
        tabu_ts.append(time.time() - t0)
        tabu_fs.append(F_t)
        log_fn(f"  ІЗ #{seed+1}: Жадібний F={F_g} ({greedy_ts[-1]:.3f}с), "
               f"Табу F={F_t} ({tabu_ts[-1]:.3f}с)")

    avg_improve = (
        (np.mean(tabu_fs) - np.mean(greedy_fs)) / np.mean(greedy_fs) * 100
        if np.mean(greedy_fs) > 0 else 0.0
    )
    log_fn(f"  Середнє покращення табу: {avg_improve:+.1f}%")
    return greedy_fs, tabu_fs, greedy_ts, tabu_ts


def run_budget_experiment(m, n, alpha, beta, h, lambda_lim,
                           runs, d, max_iter, tabu_size, log_fn):
    """Дослідження впливу бюджету L."""
    budgets = [int(alpha * k) for k in [1, 2, 3, 5, 7, 10]]
    greedy_fs, tabu_fs = [], []

    log_fn("\nЕксперимент: вплив бюджету L")
    for L_i in budgets:
        gf, tf = [], []
        for seed in range(runs):
            U_i, S_i, _, _, _ = generate_problem(m, n, alpha, beta, d=d, seed=seed)
            _, _, F_g, _, _ = greedy_algorithm(
                U_i, S_i, m, n, alpha, beta, L_i, h, lambda_lim)
            _, _, F_t, _, _ = tabu_search(
                U_i, S_i, m, n, alpha, beta, L_i, h, lambda_lim,
                max_iter=max_iter, tabu_size=tabu_size)
            gf.append(F_g)
            tf.append(F_t)
        greedy_fs.append(np.mean(gf))
        tabu_fs.append(np.mean(tf))
        log_fn(f"  L={L_i}: Жадібний={greedy_fs[-1]:.1f}, Табу={tabu_fs[-1]:.1f}")

    return budgets, greedy_fs, tabu_fs
