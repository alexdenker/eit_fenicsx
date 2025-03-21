"""

Author: derick@uni-bremen.de

"""

import numpy as np
from numpy.typing import NDArray


def add_texture(x, y, kx, ky, angle, centre):
    x_rot = centre[0] + x * np.cos(angle) - y * np.sin(angle)
    y_rot = centre[1] + x * np.sin(angle) + y * np.cos(angle)
    return 0.5 * (np.sin(kx * x_rot) + np.sin(ky * y_rot))


def cart_ellipse(x, y, h, k, a, b, alpha):
    L = ((x - h) * np.cos(alpha) + (y - k) * np.sin(alpha)) ** 2 / (a**2)
    R = ((x - h) * np.sin(alpha) - (y - k) * np.cos(alpha)) ** 2 / (b**2)
    return L + R


def sample_ellipse(test_step: int = 200, tolerance: int = 50):
    h = np.random.rand() * 1.7 - 0.85
    k = np.random.rand() * 1.7 - 0.85
    alpha = np.random.rand() * 2 * np.pi
    a = np.random.rand() * 0.8 + 0.2
    b = np.random.rand() * (a - 0.1) + 0.2

    theta = np.linspace(0, 2 * np.pi, test_step)
    x = h + a * np.cos(alpha) * np.cos(theta) - b * np.sin(alpha) * np.sin(theta)
    y = k + a * np.sin(alpha) * np.cos(theta) + b * np.cos(alpha) * np.sin(theta)

    i = 0
    while np.any(x**2 + y**2 > 0.9):
        b = np.random.rand() * (a - 0.1) + 0.2
        b = np.random.rand() * 0.9 + 0.1
        x = h + a * np.cos(alpha) * np.cos(theta) - b * np.sin(alpha) * np.sin(theta)
        y = k + a * np.sin(alpha) * np.cos(theta) + b * np.cos(alpha) * np.sin(theta)
        if i == tolerance:
            return sample_ellipse(test_step)
        i += 1

    return h, k, a, b, alpha


def sample_inclusions(numInc: int, test_step: int = 200, tolerance: int = 30):
    h, k, a, b, alpha = (
        np.zeros(numInc),
        np.zeros(numInc),
        np.zeros(numInc),
        np.zeros(numInc),
        np.zeros(numInc),
    )

    h[0], k[0], a[0], b[0], alpha[0] = sample_ellipse(test_step)

    for i in range(1, numInc):
        overlap = True
        tol = 0
        while overlap and tol < tolerance:
            overlap = False
            h[i], k[i], a[i], b[i], alpha[i] = sample_ellipse(test_step)
            for j in range(i):
                if np.linalg.norm([h[j] - h[i], k[j] - k[i]]) < a[j] + a[i] + 0.1:
                    overlap = True
                    break
            tol += 1
        if tol == tolerance:
            return sample_inclusions(numInc, test_step, tolerance)

    return h, k, a, b, alpha


def gen_conductivity(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    max_numInc: int,
    texture: bool = False,
    backCond: float = 1.0,
) -> NDArray[np.float64]:
    # (mesh, max_numInc, texture):
    # nodes = mesh['Nodes']

    # x1 = nodes[0, :]
    # x2 = nodes[1, :]

    numInc = np.random.randint(1, max_numInc + 1)

    cond = np.zeros(numInc)
    kx = np.zeros(numInc)
    ky = np.zeros(numInc)

    condOut = np.ones(x1.shape) * backCond

    h, k, a, b, alpha = sample_inclusions(numInc)

    if isinstance(texture, bool):
        if texture:
            for i in range(numInc):
                X = cart_ellipse(x1, x2, h[i], k[i], a[i], b[i], alpha[i])
                X1 = x1[X <= 1]
                X2 = x2[X <= 1]
                kx[i] = np.random.uniform(5, 15)
                ky[i] = np.random.uniform(5, 15)
                centre = [h[i], k[i]]
                res = 0.5 * (1 + add_texture(X2, X1, kx[i], ky[i], alpha[i], centre))

                cond_opt = [0, 1]
                cond_idx = np.random.randint(2)
                cond[i] = cond_opt[cond_idx]

                res = 0.6 * res + 0.2 + cond[i] * (0.2 * res + 1)
                condOut[X <= 1] = res
        else:
            for i in range(numInc):
                cond_opt = [
                    np.random.rand() * 0.29 + 0.01,
                    np.random.rand() * 1.0 + 2.0,
                ]
                cond_idx = np.random.randint(2)
                cond[i] = cond_opt[cond_idx]

                X = cart_ellipse(x1, x2, h[i], k[i], a[i], b[i], alpha[i])
                condOut[X <= 1] = cond[i]
    else:
        print("\nTexture was neither true or false")
        print("\ndefaulting to case of constant conductivity of 1\n\n")

    return condOut
