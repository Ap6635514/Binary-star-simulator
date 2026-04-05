import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# ==============================
# IMPORT FROM PACKAGE
# ==============================
from binary_star_simulator.presets import equal_mass, unequal_mass
from binary_star_simulator.utils import compute_energy, compute_barycenter
from binary_star_simulator.physics import rk4_step
from binary_star_simulator.config import G, dt, steps

# ==============================
# CLI ARGUMENT
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--preset", type=str, default="unequal")
args = parser.parse_args()

if args.preset == "equal":
    preset = equal_mass()
else:
    preset = unequal_mass()

m1, m2 = preset["m1"], preset["m2"]
r1, v1 = preset["r1"], preset["v1"]
r2, v2 = preset["r2"], preset["v2"]

# ==============================
# SIMULATION
# ==============================
positions1 = []
positions2 = []
energies = []

r1_curr, v1_curr = r1.copy(), v1.copy()
r2_curr, v2_curr = r2.copy(), v2.copy()

for _ in range(steps):
    positions1.append(r1_curr.copy())
    positions2.append(r2_curr.copy())

    # Energy tracking
    energy = compute_energy(r1_curr, v1_curr, r2_curr, v2_curr, m1, m2, G)
    energies.append(energy)

    # RK4 update
    r1_curr, v1_curr, r2_curr, v2_curr = rk4_step(
        r1_curr, v1_curr, r2_curr, v2_curr, m1, m2, G, dt
    )

positions1 = np.array(positions1)
positions2 = np.array(positions2)

# ==============================
# BARYCENTER
# ==============================
barycenter = np.array([
    compute_barycenter(r1, r2, m1, m2)
    for r1, r2 in zip(positions1, positions2)
])

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
# SAVE GIF
# ==============================
print("Saving GIF... please wait ⏳")
anim.save("binary_star.gif", writer="pillow", fps=30)

# ==============================
# ENERGY GRAPH
# ==============================
plt.figure()
plt.plot(energies)
plt.title("Total Energy Over Time")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.grid()

plt.show()