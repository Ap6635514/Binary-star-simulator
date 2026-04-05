import numpy as np

def compute_energy(r1, v1, r2, v2, m1, m2, G):
    KE = 0.5 * m1 * np.linalg.norm(v1)**2 + 0.5 * m2 * np.linalg.norm(v2)**2
    r = np.linalg.norm(r2 - r1)
    PE = -G * m1 * m2 / r
    return KE + PE


def compute_barycenter(r1, r2, m1, m2):
    return (m1 * r1 + m2 * r2) / (m1 + m2)