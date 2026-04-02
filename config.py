import numpy as np

# Gravitational constant (scaled for simulation)

G = 1.0

# Time step

dt = 0.01

# Number of simulation steps

steps = 5000
# Star 1

m1 = 1.0
r1 = np.array([-0.5, 0.0])
v1 = np.array([0.0, -0.8])

# Star 2

m2 = 1.0
r2 = np.array([0.5, 0.0])
v2 = np.array([0.0, 0.8])
