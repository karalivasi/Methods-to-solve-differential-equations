import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return (y**2 - x*y) / x**2

def runge_kutta_4(f, x0, y0, b, n):
    h = (b - x0) / n
    x = [x0]
    y = [y0]

    for i in range(n):
        x_i = x[-1]
        y_i = y[-1]
        k1 = f(x_i, y_i)
        k2 = f(x_i + h / 2, y_i + h / 2 * k1)
        k3 = f(x_i + h / 2, y_i + h / 2 * k2)
        k4 = f(x_i + h, y_i + h * k3)
        y_i = y_i + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_i = x_i + h

        x.append(x_i)
        y.append(y_i)

    return x, y


x0 = 0.5
y0 = 0.8
b = 2.5
n = 20

x_i, y_i = runge_kutta_4(f, x0, y0, b, n)

print('Решение методом Рунге-Кутта 4-го порядка:')
print('x\ty')
for i in range(len(x_i)):
    print(f'{x_i[i]:.2f}\t{y_i[i]:.6f}')

plt.figure(figsize=(12, 7))

# График численного решения
plt.plot(x_i, y_i, 'bo-', linewidth=2, markersize=6,
         markerfacecolor='red', markeredgecolor='blue',
         label=f'Рунге-Кутта 4-го порядка (n={n})')

# Подписи для некоторых ключевых точек
key_points = [0, 5, 10, 15, 20]  # Показываем каждую 5-ю точку
for i in key_points:
    if i < len(x_i):
        plt.annotate(f'({x_i[i]:.2f}, {y_i[i]:.4f})',
                     (x_i[i], y_i[i]),
                     textcoords="offset points",
                     xytext=(0, 12),
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Настройки графика
plt.xlabel('x', fontsize=14, fontweight='bold')
plt.ylabel('y(x)', fontsize=14, fontweight='bold')
plt.title(f'Решение ОДУ методом Рунге-Кутта 4-го порядка\n'
          f'$y\' = \\frac{{y^2 - xy}}{{x^2}}$, $y({x0})={y0}$, $n={n}$',
          fontsize=16, fontweight='bold')

# Сетка
plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)
plt.minorticks_on()
plt.grid(which='minor', alpha=0.2)

# Оси
plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)

# Настройки диапазона
plt.xlim(x0 - 0.1, b + 0.1)
plt.ylim(min(y_i) - 0.01, max(y_i) + 0.01)

# Легенда
plt.legend(fontsize=12, loc='upper right', framealpha=0.9)

# Информация на графике
info_text = f'Параметры:\n'
info_text += f'• Начальная точка: ({x0}, {y0})\n'
info_text += f'• Конечная точка: x = {b}\n'
info_text += f'• Количество шагов: n = {n}\n'
info_text += f'• Шаг: h = {(b-x0)/n:.4f}\n'
info_text += f'• y({b}) ≈ {y_i[-1]:.6f}'

plt.text(0.02, 0.98, info_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout()

# Сохранение графика в файл
plt.savefig('runge_kutta_4th_order_n20.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен как 'runge_kutta_4th_order_n20.png'")

plt.show()

