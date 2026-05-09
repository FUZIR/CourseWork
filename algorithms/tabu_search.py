"""
Алгоритм табуйованого пошуку для розміщення датчиків моніторингу довкілля.

Ідея: ітеративне покращення початкового (жадібного) розв'язку через дослідження
сусідів (зсув, зміна типу, додавання, видалення датчика). Список табу блокує
нещодавно виконані ходи; аспіраційний критерій дозволяє таб-хід, якщо він
покращує глобальний рекорд.

Складність: O(MaxIter × mn)
"""

import random
from collections import deque

import numpy as np

from core.utils import compute_coverage_matrix, compute_objective
from algorithms.greedy import greedy_algorithm


def tabu_search(U, S, m, n, alpha, beta, L, h, lambda_lim,
                max_iter=150, tabu_size=10,
                init_A=None, init_B=None):
    """
    Параметри
    ---------
    U, S, m, n, alpha, beta, L, h, lambda_lim : параметри задачі
    max_iter   : максимальна кількість ітерацій (критерій зупинки)
    tabu_size  : розмір списку табу (Tsize)
    init_A, init_B : початковий розв'язок (якщо None — використовується жадібний)

    Повертає
    --------
    best_A, best_B : координати датчиків найкращого знайденого розв'язку
    best_F         : значення ЦФ найкращого розв'язку
    best_K         : матриця покриття найкращого розв'язку
    history        : список dict {iter, F, best_F} для кожної ітерації
    """
    # ---- Ініціалізація початкового розв'язку ----
    if init_A is None or init_B is None:
        curr_A, curr_B, _, curr_K, _ = greedy_algorithm(
            U, S, m, n, alpha, beta, L, h, lambda_lim)
    else:
        curr_A = list(init_A)
        curr_B = list(init_B)
        curr_K = compute_coverage_matrix(curr_A, curr_B, m, n)

    curr_F = compute_objective(U, curr_K)
    best_A, best_B = list(curr_A), list(curr_B)
    best_F = curr_F
    best_K = curr_K.copy()

    tabu_list = deque(maxlen=tabu_size)
    history = [{'iter': 0, 'F': curr_F, 'best_F': best_F}]

    for iteration in range(1, max_iter + 1):
        all_placed = set(map(tuple, curr_A)) | set(map(tuple, curr_B))
        free_cells = [
            (i, j) for i in range(m) for j in range(n)
            if S[i][j] == 1 and (i, j) not in all_placed
        ]

        neighbors = _generate_neighbors(
            curr_A, curr_B, all_placed, free_cells,
            S, m, n, alpha, beta, L, h)

        step_best = _select_best_neighbor(
            neighbors, U, m, n, lambda_lim, tabu_list, best_F)

        if step_best is None:
            history.append({'iter': iteration, 'F': curr_F, 'best_F': best_F})
            break

        curr_A, curr_B, curr_K, curr_F, best_move = step_best
        tabu_list.append(best_move)

        if curr_F > best_F:
            best_F = curr_F
            best_A = list(curr_A)
            best_B = list(curr_B)
            best_K = curr_K.copy()

        history.append({'iter': iteration, 'F': curr_F, 'best_F': best_F})

    return best_A, best_B, best_F, best_K, history


# ----------------------------------------------------------------
# Внутрішні функції
# ----------------------------------------------------------------

def _generate_neighbors(curr_A, curr_B, all_placed, free_cells,
                         S, m, n, alpha, beta, L, h):
    """Генерує список сусідніх розв'язків (new_A, new_B, move)."""
    neighbors = []
    cost_now = len(curr_A) * alpha + len(curr_B) * beta

    # 1. Зсув датчика A у сусідні клітинки (радіус 2)
    for idx, (si, sj) in enumerate(curr_A):
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                ni, nj = si + di, sj + dj
                if 0 <= ni < m and 0 <= nj < n and S[ni][nj] == 1 \
                        and (ni, nj) not in all_placed:
                    new_A = [p for k, p in enumerate(curr_A) if k != idx] + [(ni, nj)]
                    neighbors.append((new_A, list(curr_B), ('move_A', (si, sj), (ni, nj))))

    # 2. Зсув датчика B у сусідні клітинки (радіус 2)
    for idx, (si, sj) in enumerate(curr_B):
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                ni, nj = si + di, sj + dj
                if 0 <= ni < m and 0 <= nj < n and S[ni][nj] == 1 \
                        and (ni, nj) not in all_placed:
                    new_B = [p for k, p in enumerate(curr_B) if k != idx] + [(ni, nj)]
                    neighbors.append((list(curr_A), new_B, ('move_B', (si, sj), (ni, nj))))

    # 3. Зміна типу A → B
    if len(curr_B) < h:
        for idx, pos in enumerate(curr_A):
            if cost_now + (beta - alpha) <= L:
                new_A = [p for k, p in enumerate(curr_A) if k != idx]
                neighbors.append((new_A, list(curr_B) + [pos], ('swap_AB', pos, pos)))

    # 4. Зміна типу B → A
    for idx, pos in enumerate(curr_B):
        new_B = [p for k, p in enumerate(curr_B) if k != idx]
        neighbors.append((list(curr_A) + [pos], new_B, ('swap_BA', pos, pos)))

    # 5. Додавання датчика (якщо є бюджет)
    for (ni, nj) in random.sample(free_cells, min(8, len(free_cells))):
        if cost_now + alpha <= L:
            neighbors.append((list(curr_A) + [(ni, nj)], list(curr_B),
                              ('add_A', None, (ni, nj))))
        if cost_now + beta <= L and len(curr_B) < h:
            neighbors.append((list(curr_A), list(curr_B) + [(ni, nj)],
                              ('add_B', None, (ni, nj))))

    # 6. Видалення датчика
    for idx, pos in enumerate(curr_A):
        neighbors.append(([p for k, p in enumerate(curr_A) if k != idx],
                          list(curr_B), ('remove_A', pos, None)))
    for idx, pos in enumerate(curr_B):
        neighbors.append((list(curr_A),
                          [p for k, p in enumerate(curr_B) if k != idx],
                          ('remove_B', pos, None)))

    return neighbors


def _select_best_neighbor(neighbors, U, m, n, lambda_lim, tabu_list, best_F):
    """
    Перебирає сусідів, перевіряє обмеження та повертає найкращий допустимий.
    Повертає (new_A, new_B, new_K, new_F, move) або None.
    """
    step_best_F = -1
    result = None

    for (new_A, new_B, move) in neighbors:
        # Перевірка унікальності
        all_pos = set(map(tuple, new_A)) | set(map(tuple, new_B))
        if len(all_pos) != len(new_A) + len(new_B):
            continue

        new_K = compute_coverage_matrix(new_A, new_B, m, n)
        if np.any(new_K > lambda_lim):
            continue

        new_F = compute_objective(U, new_K)

        # Перевірка списку табу (аспіраційний критерій)
        if move in tabu_list and new_F <= best_F:
            continue

        if new_F > step_best_F:
            step_best_F = new_F
            result = (new_A, new_B, new_K, new_F, move)

    return result
