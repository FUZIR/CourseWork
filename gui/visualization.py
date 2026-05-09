"""
Функції візуалізації: карта розміщення датчиків, графік збіжності, матриці.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.utils import SENSOR_A_RADIUS, SENSOR_B_RADIUS


def draw_map(fig, U, S, A, B, K, title):
    """
    Малює карту розміщення датчиків із зонами покриття.
    fig.clear() викликається всередині, щоб colorbar не накопичувався.
    """
    m, n = U.shape
    fig.clear()
    ax = fig.add_subplot(111)

    # Теплова карта ваг
    im = ax.imshow(U, cmap='YlOrRd', alpha=0.55, interpolation='nearest',
                   extent=[-0.5, n - 0.5, m - 0.5, -0.5], vmin=0)
    fig.colorbar(im, ax=ax, label='Вага ділянки', fraction=0.04, pad=0.02)

    # Сітка
    for i in range(m + 1):
        ax.axhline(i - 0.5, color='#888888', linewidth=0.4)
    for j in range(n + 1):
        ax.axvline(j - 0.5, color='#888888', linewidth=0.4)

    # Заборонені клітинки
    for i in range(m):
        for j in range(n):
            if S[i][j] == 0:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, color='#333333', alpha=0.7, zorder=2))
                ax.text(j, i, 'X', ha='center', va='center',
                        color='white', fontsize=7, zorder=3)

    # Зони покриття
    if K is not None:
        for i in range(m):
            for j in range(n):
                if K[i][j] > 0:
                    alpha_val = min(0.45, 0.15 * K[i][j])
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        color='#1565C0', alpha=alpha_val, zorder=3))

    placed_A = set(A)
    placed_B = set(B)

    # Датчики типу A
    for (i, j) in A:
        r = SENSOR_A_RADIUS
        ax.add_patch(plt.Rectangle(
            (j - r - 0.5, i - r - 0.5), 2 * r + 1, 2 * r + 1,
            fill=False, edgecolor='#1976D2', linewidth=2.2, zorder=4))
        ax.plot(j, i, 's', color='#1976D2', markersize=14, zorder=5,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.text(j, i, 'A', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold', zorder=6)

    # Датчики типу B
    for (i, j) in B:
        r = SENSOR_B_RADIUS
        ax.add_patch(plt.Rectangle(
            (j - r - 0.5, i - r - 0.5), 2 * r + 1, 2 * r + 1,
            fill=False, edgecolor='#D32F2F', linewidth=2.2,
            linestyle='--', zorder=4))
        ax.plot(j, i, 's', color='#D32F2F', markersize=14, zorder=5,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.text(j, i, 'B', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold', zorder=6)

    # Числові ваги у вільних клітинках
    for i in range(m):
        for j in range(n):
            if S[i][j] == 1 and (i, j) not in placed_A and (i, j) not in placed_B:
                ax.text(j, i, str(int(U[i][j])),
                        ha='center', va='center', fontsize=7, color='#212121', zorder=4)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(j + 1) for j in range(n)])
    ax.set_yticks(range(m))
    ax.set_yticklabels([str(i + 1) for i in range(m)])
    ax.set_xlabel("Стовпець j", fontsize=10)
    ax.set_ylabel("Рядок i", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')

    legend_el = [
        mpatches.Patch(facecolor='#1976D2', edgecolor='white', label='Датчик A (3×3)'),
        mpatches.Patch(facecolor='#D32F2F', edgecolor='white', label='Датчик B (5×5)'),
        mpatches.Patch(color='#1565C0', alpha=0.35, label='Зона покриття'),
        mpatches.Patch(color='#333333', alpha=0.7,  label='Заборонена зона'),
    ]
    ax.legend(handles=legend_el, loc='upper right', fontsize=8, framealpha=0.85)
    fig.tight_layout()


def draw_convergence(fig, ax, curr_f, best_f, title):
    """Графік збіжності алгоритму."""
    ax.clear()
    iters = range(len(curr_f))
    ax.plot(iters, curr_f, color='#1976D2', linewidth=1.2, alpha=0.7,
            label='Поточне F')
    ax.plot(iters, best_f, color='#D32F2F', linewidth=2.2,
            label='Найкраще F (рекорд)')
    ax.fill_between(iters, curr_f, best_f, alpha=0.1, color='#D32F2F')
    ax.set_xlabel("Ітерація", fontsize=10)
    ax.set_ylabel("Цільова функція F", fontsize=10)
    ax.set_title(f"Збіжність: {title}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def draw_matrices(fig, U, S):
    """Відображає матриці U та S."""
    m, n = U.shape
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    im1 = ax1.imshow(U, cmap='YlOrRd', interpolation='nearest')
    fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.04)
    ax1.set_title("Матриця ваг U", fontsize=11, fontweight='bold')
    ax1.set_xlabel("j"); ax1.set_ylabel("i")
    ax1.set_xticks(range(n)); ax1.set_yticks(range(m))
    ax1.set_xticklabels(range(1, n + 1)); ax1.set_yticklabels(range(1, m + 1))
    for i in range(m):
        for j in range(n):
            ax1.text(j, i, str(int(U[i][j])),
                     ha='center', va='center', fontsize=7,
                     color='black' if U[i][j] < 35 else 'white')

    ax2.imshow(S, cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title("Матриця доступності S", fontsize=11, fontweight='bold')
    ax2.set_xlabel("j"); ax2.set_ylabel("i")
    ax2.set_xticks(range(n)); ax2.set_yticks(range(m))
    ax2.set_xticklabels(range(1, n + 1)); ax2.set_yticklabels(range(1, m + 1))
    for i in range(m):
        for j in range(n):
            val = "+" if S[i][j] == 1 else "X"
            ax2.text(j, i, val, ha='center', va='center', fontsize=9,
                     color='white' if S[i][j] == 0 else 'black',
                     fontweight='bold')

    fig.tight_layout()


def draw_comparison_bars(fig, f_vals, t_vals):
    """Стовпчаста діаграма порівняння алгоритмів (F та час)."""
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    labels = ['Жадібний', 'Табу-пошук']
    colors = ['#2196F3', '#FF5722']

    bars1 = ax1.bar(labels, f_vals, color=colors, width=0.5, edgecolor='black')
    ax1.set_title("Цільова функція F"); ax1.set_ylabel("F")
    for bar, val in zip(bars1, f_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    bars2 = ax2.bar(labels, t_vals, color=colors, width=0.5, edgecolor='black')
    ax2.set_title("Час виконання (с)"); ax2.set_ylabel("час, с")
    for bar, val in zip(bars2, t_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.4f}", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
