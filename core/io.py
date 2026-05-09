"""
Збереження та завантаження екземплярів задачі у форматі JSON.
"""

import json
import numpy as np


def save_problem(path, U, S, params):
    """
    Зберігає задачу у JSON-файл.

    params : dict з ключами alpha, beta, L, h, lambda_lim
    """
    data = {
        'U': U.tolist(),
        'S': S.tolist(),
        'params': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                   for k, v in params.items()},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_problem(path):
    """
    Завантажує задачу з JSON-файлу.

    Повертає (U, S, params) або кидає виключення при помилці.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    U = np.array(data['U'], dtype=float)
    S = np.array(data['S'], dtype=int)
    return U, S, data['params']
