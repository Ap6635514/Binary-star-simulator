import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# CONFIG
# ==============================
G = 1.0
dt = 0.01
steps = 1000   # reduced for faster GIF

m1 = 2.0
m2 = 0.5

r1 = np.array([-0.2, 0.0])
v1 = np.array([0.0, -0.3])

r2 = np.array([0.8, 0.0])
v2 = np.array([0.0, 1.2])

# ==============================
# PHYSICS
# ==============================

def compute_acceleration(r1, r2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    force_dir = r / dist
    force = G * m1 * m2 / dist**2

    a1 = force_dir * force / m1
    a2 = -force_dir * force / m2

    return a1, a2


def rk4_step(r1, v1, r2, v2):

    def derivatives(r1, v1, r2, v2):
        a1, a2 = compute_acceleration(r1, r2)
        return v1, a1, v2, a2

    k1_v1, k1_a1, k1_v2, k1_a2 = derivatives(r1, v1, r2, v2)

    k2_v1, k2_a1, k2_v2, k2_a2 = derivatives(
        r1 + 0.5*dt*k1_v1,
        v1 + 0.5*dt*k1_a1,
        r2 + 0.5*dt*k1_v2,
        v2 + 0.5*dt*k1_a2
    )

    k3_v1, k3_a1, k3_v2, k3_a2 = derivatives(
        r1 + 0.5*dt*k2_v1,
        v1 + 0.5*dt*k2_a1,
        r2 + 0.5*dt*k2_v2,
        v2 + 0.5*dt*k2_a2
    )

    k4_v1, k4_a1, k4_v2, k4_a2 = derivatives(
        r1 + dt*k3_v1,
        v1 + dt*k3_a1,
        r2 + dt*k3_v2,
        v2 + dt*k3_a2
    )

    r1_new = r1 + dt*(k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)/6
    v1_new = v1 + dt*(k1_a1 + 2*k2_a1 + 2*k3_a1 + k4_a1)/6

    r2_new = r2 + dt*(k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)/6
    v2_new = v2 + dt*(k1_a2 + 2*k2_a2 + 2*k3_a2 + k4_a2)/6

    return r1_new, v1_new, r2_new, v2_new

# ==============================
# SIMULATION
# ==============================

positions1 = []
positions2 = []

r1_curr, v1_curr = r1.copy(), v1.copy()
r2_curr, v2_curr = r2.copy(), v2.copy()

for _ in range(steps):
    positions1.append(r1_curr.copy())
    positions2.append(r2_curr.copy())

    r1_curr, v1_curr, r2_curr, v2_curr = rk4_step(
        r1_curr, v1_curr, r2_curr, v2_curr
    )

positions1 = np.array(positions1)
positions2 = np.array(positions2)

# ==============================
# BARYCENTER
# ==============================

barycenter = (m1 * positions1 + m2 * positions2) / (m1 + m2)

# ==============================
# ANIMATION
# ==============================

fig, ax = plt.subplots(figsize=(6,6))
ax.set_facecolor("black")

line1, = ax.plot([], [], lw=1, color="cyan", alpha=0.6)
line2, = ax.plot([], [], lw=1, color="orange", alpha=0.6)

star1, = ax.plot([], [], 'o', color="cyan", markersize=8)
star2, = ax.plot([], [], 'o', color="orange", markersize=5)

bary, = ax.plot([], [], 'x', color="white", markersize=6)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Binary Star System", color="white")
ax.set_aspect('equal')
ax.grid(color='gray', linestyle='--', alpha=0.3)
ax.tick_params(colors='white')

def update(frame):
    line1.set_data(positions1[:frame,0], positions1[:frame,1])
    line2.set_data(positions2[:frame,0], positions2[:frame,1])

    star1.set_data([positions1[frame,0]], [positions1[frame,1]])
    star2.set_data([positions2[frame,0]], [positions2[frame,1]])

    bary.set_data([barycenter[frame,0]], [barycenter[frame,1]])

    return line1, line2, star1, star2, bary

anim = FuncAnimation(fig, update, frames=steps, interval=10)

# ==============================
# SAVE + SHOW
# ==============================

anim.save("binary_star.gif", writer="pillow", fps=30)

plt.show()
