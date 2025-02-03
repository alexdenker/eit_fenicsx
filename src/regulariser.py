"""
Code to construct:
    - Smoothness Regulariser


"""

import numpy as np
import matplotlib.pyplot as plt

from dolfinx.fem import Function, functionspace

from tqdm import tqdm


def create_smoothness_regulariser(
    omega, save_name: str, corrlength: float = 1.0, std: float = 0.3
):
    """
    Gaussian smoothness prior with covariance
        Gamma = std**2 exp(|| xi - xj ||^2 / (2 * corrlength**2))

    From: https://zenodo.org/record/8252370

    Returns: L with L.T @ L = Gamma^(-1)

    """

    var = std**2
    c = 1e-7

    V = functionspace(omega, ("DG", 0))

    u_sol = Function(V)
    dofs = len(u_sol.x.array)

    mesh_pos = np.array(V.tabulate_dof_coordinates()[:, :2])

    g = mesh_pos

    ng = dofs
    a = var - c
    b = corrlength

    Gamma_pr = np.zeros((ng, ng))

    for ii in tqdm(range(ng)):
        for jj in range(ii, ng):
            dist_ij = np.linalg.norm(g[ii, :] - g[jj, :])
            gamma_ij = a * np.exp(-(dist_ij**2) / (2 * b**2))

            if ii == jj:
                gamma_ij = gamma_ij + c
            Gamma_pr[ii, jj] = gamma_ij
            Gamma_pr[jj, ii] = gamma_ij

    L = np.linalg.cholesky(np.linalg.inv(Gamma_pr)).T

    np.save(save_name, L)


def plot_samples_from_prior(L, triangulation):
    """
    L: Cholesky decomposition of precision matrix, L.T @ L = Gamma^(-1)
    triangulation: numpy triangulation of the underlying mesh

    """

    samples = np.linalg.solve(L, np.random.randn(L.shape[0], 4))
    print(samples.shape)
    fig, axes = plt.subplots(1, 4, figsize=(14, 6))

    for i in range(4):
        im = axes[i].tripcolor(
            triangulation, samples[:, i].flatten(), cmap="jet", shading="flat"
        )
        axes[i].axis("image")
        axes[i].set_aspect("equal", adjustable="box")
        axes[i].set_title("Sample " + str(i))
        fig.colorbar(im, ax=axes[i])

    return fig
