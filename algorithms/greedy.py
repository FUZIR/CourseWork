"""
Жадібний алгоритм розміщення датчиків моніторингу довкілля.

Ідея: на кожній ітерації обирається позиція та тип датчика з максимальною
питомою ефективністю ΔF / cost, з урахуванням усіх обмежень задачі.

Складність: O(m²n²)
"""

import numpy as np
from core.utils import compute_delta_f, compute_objective


def greedy_algorithm(U, S, m, n, alpha, beta, L, h, lambda_lim):
    """
    Параметри
    ---------
    U : матриця ваг (m×n)
    S : матриця доступності (m×n, 0/1)
    m, n       : розмір сітки
    alpha, beta: вартість датчиків A та B
    L : бюджет
    h          : максимальна кількість датчиків типу B
    lambda_lim : ліміт перекриття ділянки

    Повертає
    --------
    A : список координат датчиків типу A
    B : список координат датчиків типу B
    F : значення цільової функції
    K : матриця кратності покриття
    history : список dict з інформацією про кожну ітерацію
    """
    A = []
    B = []
    K = np.zeros((m, n), dtype=int)
    placed = set()
    L_rem = L
    h_rem = h
    history = []

    for iteration in range(1, m * n + 1):
        E_max = -1.0
        best_pos = None
        best_type = None
        best_delta = 0
        best_cells = None

        for i in range(m):
            for j in range(n):
                if S[i][j] == 0 or (i, j) in placed:
                    continue

                if L_rem >= alpha:
                    delta, cells = compute_delta_f(i, j, 'A', U, K, lambda_lim, m, n)
                    if delta is not None and delta > 0:
                        E = delta / alpha
                        if E > E_max:
                            E_max, best_pos, best_type = E, (i, j), 'A'
                            best_delta, best_cells = delta, cells

                if L_rem >= beta and h_rem > 0:
                    delta, cells = compute_delta_f(i, j, 'B', U, K, lambda_lim, m, n)
                    if delta is not None and delta > 0:
                        E = delta / beta
                        if E > E_max:
                            E_max, best_pos, best_type = E, (i, j), 'B'
                            best_delta, best_cells = delta, cells

        if best_pos is None or E_max <= 0:
            break



        if best_type == 'A':
            A.append(best_pos)
            L_rem -= alpha
        else:
            B.append(best_pos)
            L_rem -= beta
            h_rem -= 1

        placed.add(best_pos)
        for (ci, cj) in best_cells:
            K[ci][cj] += 1

        F = compute_objective(U, K)
        history.append({
            'iter':  iteration,
            'pos':   best_pos,
            'type':  best_type,
            'delta': best_delta,
            'F':     F,
            'L_rem': L_rem,
        })

        if L_rem < alpha and (L_rem < beta or h_rem == 0):
            break

    return A, B, compute_objective(U, K), K, history

