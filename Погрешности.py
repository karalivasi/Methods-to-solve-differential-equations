import numpy as np
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

    for i in range(n):
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

y_anal_i = analitic(np.array(x_i))
y_anal_j = analitic(np.array(x_j))
y_anal_k = analitic(np.array(x_k))


abs_error_euler = np.abs(y_anal_i - np.array(y_i))
abs_error_heun = np.abs(y_anal_j - np.array(y_j))
abs_error_rk4 = np.abs(y_anal_k - np.array(y_k))

print("Абсолютные погрешности по шагам:")
print(f"{'k':<3} {'x':<6} {'Эйлер':<10} {'Хойн':<10} {'РК4':<10}")
print("-" * 50)

for k in range(len(x_i)):
    print(f"{k:<3} {x_i[k]:<6.2f} {abs_error_euler[k]:<10.6f} "
          f"{abs_error_heun[k]:<10.6f} {abs_error_rk4[k]:<10.6f}")
rel_error_euler = abs_error_euler / np.abs(y_i)
rel_error_heun = abs_error_heun / np.abs(y_j)
rel_error_rk4 = abs_error_rk4 / np.abs(y_k)

print("Относительные погрешности по шагам:")
print(f"{'k':<3} {'x':<6} {'Эйлер':<10} {'Хойн':<10} {'РК4':<10}")
print("-" * 50)

for k in range(len(x_i)):
    print(f"{k:<3} {x_i[k]:<6.2f} {rel_error_euler[k]:<10.6f} "
          f"{rel_error_heun[k]:<10.6f} {rel_error_rk4[k]:<10.6f}")

max_abs_euler = np.max(abs_error_euler)
max_abs_heun = np.max(abs_error_heun)
max_abs_rk4 = np.max(abs_error_rk4)

max_rel_euler = np.max(rel_error_euler)
max_rel_heun = np.max(rel_error_heun)
max_rel_rk4 = np.max(rel_error_rk4)

print("Максимальные абсолютные погрешности:")
print(f"Эйлер: Δ = {max_abs_euler:.6f}")
print(f"Хойн:  Δ = {max_abs_heun:.6f}")
print(f"РК4:   Δ = {max_abs_rk4:.6f}")
print()

print("Максимальные относительные погрешности:")
print(f"Эйлер: δ = {max_rel_euler:.6f}")
print(f"Хойн:  δ = {max_rel_heun:.6f}")
print(f"РК4:   δ = {max_rel_rk4:.6f}")
print()

