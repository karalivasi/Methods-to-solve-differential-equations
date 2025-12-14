import numpy as np
import matplotlib.pyplot as plt
def f(x,y):
    return (y**2-x*y)/x**2

def euler_method(f, x0, y0, b, n):
    h = (b - x0) / n
    x = [x0]
    y = [y0]

    for i in range(n):
        x_i = x[-1] + h
        y_i = y[-1] + h * f(x[-1], y[-1])
        x.append(x_i)
        y.append(y_i)
    return x, y
x0=0.5
y0=  0.8
b, n =2.5, 20

x_i, y_i = euler_method(f, x0, y0, b, n)

def heun_method(f, x0, y0, b, n):
    h = (b - x0) / n
    x = [x0]
    y = [y0]

    for j in range(n):
        x_j = x[-1]
        y_j = y[-1]
        x_euler, y_euler = euler_method(f, x_j, y_j, x_j + h, 1)
        y_euler= y_euler[1]
        k1 = f(x_j, y_j)
        k2 = f(x_j + h, y_euler)
        y_j= y_j + h * 0.5 * (k1 + k2)
        x.append(x_j + h)
        y.append(y_j)

    return x, y


x_j, y_j = heun_method(f, x0, y0, b, n)

def runge_kutta_4(f, x0, y0, b, n):
    h = (b - x0) / n
    x = [x0]
    y = [y0]

    for k in range(n):
        x_k = x[-1]
        y_k = y[-1]
        k1 = f(x_k, y_k)
        k2 = f(x_k + h / 2, y_k + h / 2 * k1)
        k3 = f(x_k + h / 2, y_k + h / 2 * k2)
        k4 = f(x_k + h, y_k + h * k3)
        y_k = y_k + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_k = x_k + h

        x.append(x_k)
        y.append(y_k)

    return x, y

x_k, y_k = runge_kutta_4(f, x0, y0, b, n)

def analitic(x):
    return 2*x/(1+x**2)
x_analitic = np.linspace(x0, b, 200)
y_analitic = analitic(x_analitic)


plt.figure(figsize=(14, 8))
plt.plot(x_analitic, y_analitic, 'k-', linewidth=3, alpha=0.8, label='Аналитическое решение: $y = \\frac{2x}{1+x^2}$')
plt.plot(x_i, y_i, 'ro-', linewidth=1.5, markersize=6,
         markerfacecolor='red', markeredgecolor='darkred', alpha=0.8,
         label='Метод Эйлера (n=20)')
plt.plot(x_j, y_j, 'gs-', linewidth=1.5, markersize=6,
         markerfacecolor='lime', markeredgecolor='darkgreen', alpha=0.8,
         label='Метод Хойна (n=20)')
plt.plot(x_k, y_k, 'b^-', linewidth=1.5, markersize=6,
         markerfacecolor='cyan', markeredgecolor='darkblue', alpha=0.8,
         label='Метод Рунге-Кутта 4 (n=20)')
plt.plot(x0, y0, 'k*', markersize=15, label=f'Начальное условие: $y({x0})={y0}$')
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.title('Сравнение численных методов решения ОДУ\n$y\' = \\frac{y^2 - xy}{x^2}$',
          fontsize=14, pad=15)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='upper right')
plt.xlim(x0 - 0.1, b + 0.1)
plt.ylim(0.75, 1.1)
info_text = f'Параметры:\n'
info_text += f'• Интервал: [{x0}, {b}]\n'
info_text += f'• Шаги: n = {n}\n'
info_text += f'• Шаг h = {(b-x0)/n:.3f}\n'
info_text += f'• Нач.условие: y({x0}) = {y0}'

plt.text(0.02, 0.98, info_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout()
plt.show()



