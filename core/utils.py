"""
Допоміжні функції, константи, генератор задач та готові екземпляри.
"""

import random
import numpy as np

SENSOR_A_RADIUS = 1   # покриває 3×3 клітинки
SENSOR_B_RADIUS = 2   # покриває 5×5 клітинок


def get_coverage_cells(i, j, sensor_type, m, n):
    """Повертає список клітинок, що покриваються датчиком у позиції (i, j)."""
    r = SENSOR_A_RADIUS if sensor_type == 'A' else SENSOR_B_RADIUS
    return [
        (i + di, j + dj)
        for di in range(-r, r + 1)
        for dj in range(-r, r + 1)
        if 0 <= i + di < m and 0 <= j + dj < n
    ]


def compute_coverage_matrix(placements_A, placements_B, m, n):
    """Обчислює матрицю кратності покриття K з нуля."""
    K = np.zeros((m, n), dtype=int)
    for (i, j) in placements_A:
        for (ci, cj) in get_coverage_cells(i, j, 'A', m, n):
            K[ci][cj] += 1
    for (i, j) in placements_B:
        for (ci, cj) in get_coverage_cells(i, j, 'B', m, n):
            K[ci][cj] += 1
    return K


def compute_objective(U, K):
    """F = сума U[i][j] де K[i][j] >= 1."""
    return int(np.sum(U * (K >= 1)))


def check_overlap_valid(i, j, sensor_type, K, lambda_lim, m, n):
    """Перевіряє, чи не порушує встановлення датчика ліміт перекриття."""
    return all(K[ci][cj] + 1 <= lambda_lim
               for (ci, cj) in get_coverage_cells(i, j, sensor_type, m, n))


def compute_delta_f(i, j, sensor_type, U, K, lambda_lim, m, n):
    """
    Обчислює приріст ЦФ від встановлення датчика у (i, j).
    Повертає (delta_F, cells) або (None, None) якщо порушує обмеження.
    """
    if not check_overlap_valid(i, j, sensor_type, K, lambda_lim, m, n):
        return None, None
    cells = get_coverage_cells(i, j, sensor_type, m, n)
    delta = sum(U[ci][cj] for (ci, cj) in cells if K[ci][cj] == 0)
    return delta, cells


def generate_problem(m, n, alpha, beta, d=0.75, lambda_max=2, h_max=2, seed=None):
    """
    Генерує випадковий екземпляр задачі.

    Параметри
    ---------
    d          : щільність матриці S (частка доступних клітинок)
    lambda_max : верхня межа для λ
    h_max      : верхня межа для h
    seed       : seed для відтворюваності
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    U = np.random.randint(1, 51, size=(m, n)).astype(float)
    S = (np.random.random((m, n)) < d).astype(int)
    if np.sum(S) == 0:
        S[m // 2][n // 2] = 1

    num_sensors = max(2, int(d * m * n * 0.25))
    L = alpha * num_sensors + beta * min(h_max, max(1, num_sensors // 2))
    L = max(L, alpha)

    lambda_lim = random.randint(1, lambda_max)
    h = random.randint(0, h_max)

    return U, S, L, h, lambda_lim
