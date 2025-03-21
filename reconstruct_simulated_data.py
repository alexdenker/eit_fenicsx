import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import torch

from dolfinx.fem import Function, assemble_scalar, form
import ufl

from src import EIT, gen_conductivity, L1Sparsity, create_smoothness_regulariser, GaussNewtonSolver, GaussNewtonSolverTV

def compute_relative_l1_error(sigma_rec, sigma_gt):
    diff = abs(sigma_rec - sigma_gt) * ufl.dx
    diff = assemble_scalar(form(diff))

    norm = abs(sigma_gt) * ufl.dx
    norm = assemble_scalar(form(norm))

    return diff / norm


device = "cuda"

L = 16
backCond = 1.31

Injref = np.load("dataset/injection_pattern.npy")

z = 1e-6 * np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(solver.V_sigma.tabulate_dof_coordinates()[:, :2])

np.random.seed(14)  # 16 14: works well
sigma_mesh = gen_conductivity(
    mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=backCond
)
sigma_gt_vsigma = Function(solver.V_sigma)
sigma_gt_vsigma.x.array[:] = sigma_mesh

sigma_gt = Function(solver.V)
sigma_gt.interpolate(sigma_gt_vsigma)

# We simulate the measurements using our forward solver
_, U = solver.forward_solve(sigma_gt)
Umeas = np.array(U)

noise_percentage = 0.005
var_meas = (noise_percentage * np.abs(Umeas)) ** 2
delta = 0.005
Umeas = Umeas + delta * np.mean(np.abs(Umeas)) * np.random.normal(
                size=Umeas.shape
            )



l1_reconstructor = L1Sparsity(
    eit_solver=solver,
    backCond=backCond,
    kappa=0.0285,
    alpha=0.0001,
    clip=[0.01, 3.0],
    max_iter=200,
    stopping_criterion=5e-4,
    step_min=1e-6,
    initial_step_size=0.6,
)

 
sigma_reco_l1 = l1_reconstructor.forward(Umeas=Umeas)

sigma_reco_l1_vsigma = Function(solver.V_sigma)
sigma_reco_l1_vsigma.interpolate(sigma_reco_l1)


try:
    Lprior = np.load("data/L_KIT4_mesh_coarse.npy")
except FileNotFoundError:
    print("Regulariser matrix was not found. Rebuild it...this may take a while")

    create_smoothness_regulariser(
        solver.omega, "data/L_KIT4_mesh_coarse.npy", corrlength=0.2, std=0.15
    )
    Lprior = np.load("data/L_KIT4_mesh_coarse.npy")

Lprior = torch.from_numpy(Lprior).float().to(device)

R = Lprior.T @ Lprior

Umeas_flatten = np.array(Umeas).flatten()

GammaInv = 1.0 / (var_meas.flatten() + 0.001)
GammaInv = torch.from_numpy(GammaInv).float().to(device)

sigma_init = Function(solver.V_sigma)
sigma_init.x.array[:] = backCond

gauss_newton_solver = GaussNewtonSolver(solver, 
                num_steps=2,
                R=R,
                lamb=0.3,  # 8e-4,
                GammaInv=GammaInv,
                clip=[0.001, 3.0],
                backCond=backCond,
                device=device)

sigma_rec = gauss_newton_solver.forward(
    Umeas=Umeas_flatten,
    sigma_init=sigma_init,
    verbose=True
    )

sigma_init = Function(solver.V_sigma)
sigma_init.x.array[:] = backCond

gauss_newton_solver = GaussNewtonSolverTV(
    solver,
    device=device,
    num_steps=8,
    lamb=0.04,  
    beta=1e-6,  
    GammaInv=GammaInv,
    clip=[0.01, 3.0],
    backCond=backCond
)

print(gauss_newton_solver.GammaInv)

sigma_rec_tv = gauss_newton_solver.forward(
    Umeas=Umeas_flatten,
    sigma_init=sigma_init,
    verbose=True,
)


# sigma_rec_tv = Function(solver.V_sigma)
# sigma_rec_tv.x.array[:] = sigma


rel_error_l1 = compute_relative_l1_error(sigma_reco_l1_vsigma, sigma_gt_vsigma)
rel_error_gn = compute_relative_l1_error(sigma_rec, sigma_gt_vsigma)
rel_error_tv = compute_relative_l1_error(sigma_rec_tv, sigma_gt_vsigma)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(19, 6))

pred = np.array(sigma_gt_vsigma.x.array[:]).flatten()
im = ax1.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax1.axis("image")
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("GT")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.axis("off")

pred = np.array(sigma_reco_l1_vsigma.x.array[:]).flatten()
im = ax2.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax2.axis("image")
ax2.set_aspect("equal", adjustable="box")
ax2.set_title(
    f"L1-Sparsity, \n relative L1 error={np.format_float_positional(rel_error_l1,4)}"
)
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.axis("off")


pred = np.array(sigma_rec.x.array[:]).flatten()
im = ax3.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax3.axis("image")
ax3.set_aspect("equal", adjustable="box")
ax3.set_title(
    f"Gauss-Newton (Smoothness Prior), \n relative L1 error={np.format_float_positional(rel_error_gn,4)}"
)
fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
ax3.axis("off")

pred = np.array(sigma_rec_tv.x.array[:]).flatten()
im = ax4.tripcolor(tri, pred, cmap="jet", shading="flat", vmin=0.01, vmax=2.0)
ax4.axis("image")
ax4.set_aspect("equal", adjustable="box")
ax4.set_title(
    f"Gauss-Newton (TV Prior), \n relative L1 error={np.format_float_positional(rel_error_tv,4)}"
)
fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
ax4.axis("off")

plt.show()
