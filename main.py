import numpy as np
import matplotlib.pyplot as plt

# физические константы
q = -1.602176634e-19
m = 9.1093837015e-31
c = 2.99792458e8

# ввод значений E и B
Ex = float(input("Введите значение электрического поля Ex (В/м): "))
Bz = float(input("Введите значение магнитного поля Bz (Тл): "))

# начальные параметры
Bx, By = 0.0, 0.0
Ey, Ez = 0.0, 0.0
r0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([1.0e6, 0.0, 0.0])

# расчёт шага dt
B_mag = np.linalg.norm([Bx, By, Bz])
Omega_c = abs(q) * B_mag / m if B_mag > 0 else 0.0
Tc = 2.0 * np.pi / Omega_c if Omega_c > 0 else np.inf
if np.isfinite(Tc):
    dt = min(Tc / 1200.0, 5e-14)
    t_max = 4.0 * Tc
else:
    dt = 5e-14
    t_max = 2e-9

# правая часть системы
def rhs(state):
    v = state[3:6]
    v2 = np.dot(v, v)
    gamma = 1.0 / np.sqrt(1.0 - v2 / c**2)
    E = np.array([Ex, Ey, Ez])
    B = np.array([Bx, By, Bz])
    a = (q / (m * gamma)) * (E + np.cross(v, B) - (np.dot(v, E) / c**2) * v)
    return np.hstack((v, a))

# шаг Рунге–Кутты 4-го порядка
def rk4_step(state, dt):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# интегрирование
N = int(np.ceil(t_max / dt)) + 1
t = np.linspace(0.0, N * dt, N)
state = np.zeros(6)
state[0:3] = r0
state[3:6] = v0
R = np.zeros((N, 3))
V = np.zeros((N, 3))
Gamma = np.zeros(N)
R[0, :] = r0
V[0, :] = v0
Gamma[0] = 1.0 / np.sqrt(1.0 - np.dot(v0, v0) / c**2)

for i in range(1, N):
    state = rk4_step(state, dt)
    R[i, :] = state[0:3]
    V[i, :] = state[3:6]
    Gamma[i] = 1.0 / np.sqrt(1.0 - np.dot(state[3:6], state[3:6]) / c**2)

# построение графиков
plt.figure(figsize=(12, 5))

ax1 = plt.subplot(1, 2, 1)
ax1.plot(R[:, 0], R[:, 1], lw=1.2)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title(f"Релятивистская траектория (Ex={Ex:.2e} В/м, Bz={Bz:.2e} Тл)")
ax1.set_xlabel('x, м')
ax1.set_ylabel('y, м')
ax1.grid(True, ls='--', alpha=0.5)

x_min, x_max = np.min(R[:, 0]), np.max(R[:, 0])
y_min, y_max = np.min(R[:, 1]), np.max(R[:, 1])
dx, dy = x_max - x_min, y_max - y_min
x_mid, y_mid = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
base = max(dx, dy, 1e-12)
margin = 0.4 * base
ax1.set_xlim(x_mid - 0.5 * base - margin, x_mid + 0.5 * base + margin)
ax1.set_ylim(y_mid - 0.5 * base - margin, y_mid + 0.5 * base + margin)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(t, Gamma, label='γ(t)')
ax2.set_title('Фактор Лоренца')
ax2.set_xlabel('t, с')
ax2.set_ylabel('γ')
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.show()
