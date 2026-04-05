import numpy as np

def equal_mass():
    return {
        "m1": 1.0,
        "m2": 1.0,
        "r1": np.array([-0.5, 0.0]),
        "v1": np.array([0.0, -0.8]),
        "r2": np.array([0.5, 0.0]),
        "v2": np.array([0.0, 0.8]),
    }


def unequal_mass():
    return {
        "m1": 2.0,
        "m2": 0.5,
        "r1": np.array([-0.2, 0.0]),
        "v1": np.array([0.0, -0.3]),
        "r2": np.array([0.8, 0.0]),
        "v2": np.array([0.0, 1.2]),
    }


def extreme_mass():
    return {
        "m1": 5.0,
        "m2": 0.2,
        "r1": np.array([-0.1, 0.0]),
        "v1": np.array([0.0, -0.1]),
        "r2": np.array([1.0, 0.0]),
        "v2": np.array([0.0, 1.5]),
    }