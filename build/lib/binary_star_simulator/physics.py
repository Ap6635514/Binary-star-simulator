import numpy as np

def compute_acceleration(r1, r2, m1, m2, G):
    r = r2 - r1
    dist = np.linalg.norm(r)

    force_dir = r / dist
    force = G * m1 * m2 / dist**2

    a1 = force_dir * force / m1
    a2 = -force_dir * force / m2

    return a1, a2


def rk4_step(r1, v1, r2, v2, m1, m2, G, dt):

    def derivatives(r1, v1, r2, v2):
        a1, a2 = compute_acceleration(r1, r2, m1, m2, G)
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
