"""
This is an implementation of the Gauss-Newton Method with (smoothed) Total Variation regularisation as presented in 
    Borsic et al. "In Vivo Impedance Imaging With Total Variation Regularization", IEEE TMI (2009)
    https://ieeexplore.ieee.org/document/5371948

The smoothed TV regulariser is given as 
    sum_i sqrt((Li sigma)^2 + gamma) 

with a small parameter gamma > 0. 
Li = (0,...,0,1,0,...,0,-1,0,...,0) represents the discrete gradients 
and the sum goes over all edges in the mesh. 


The noise depends both on the measuremnts and the absolute maximum measurment:s
U_noisy = Umeas + (delta1 * np.abs(Umeas) + delta2 * np.max(np.abs(Umeas))) * noise

"""


import numpy as np
import os
import time
import sys

import yaml
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import torch
from omegaconf import OmegaConf

from dolfinx.fem import Function, create_nonmatching_meshes_interpolation_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eit_forward_fenicsx import EIT
from src.gauss_newton import GaussNewtonSolverTV
from src.random_ellipses import gen_conductivity
from src.utils import current_method
from src.performance_metrics import (
    RelativeL1Error,
    DiceScore,
    DynamicRange,
    MeasurementError,
    RelativeL2Error
)

import time

np.random.seed(13)

delta1 = 0.1  # noise level
delta2 = 0.001 
L = 16 # number of electrodes 
backCond = 1.0 # background conductivity 
z = 1e-6 * np.ones(L) # contact impedance 

Injref = current_method(L, L, method=2) # 1 and -1 in adjacent electrodes. 

# We set up two instances of the EIT forward operator: a coarse mesh and a dense mesh
# We use the dense mesh to simulate the measurements and the coarse mesh for reconstruction
solver_reco = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")
solver_sim = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_dense.msh")


xy = solver_reco.omega.geometry.x
cells = solver_reco.omega.geometry.dofmap.reshape((-1, solver_reco.omega.topology.dim + 1))
tri_reco = Triangulation(xy[:, 0], xy[:, 1], cells)

xy = solver_sim.omega.geometry.x
cells = solver_sim.omega.geometry.dofmap.reshape((-1, solver_sim.omega.topology.dim + 1))
tri_sim = Triangulation(xy[:, 0], xy[:, 1], cells)


reconstructor = GaussNewtonSolverTV(
    eit_solver=solver_reco,
    device="cuda",
    num_steps=10,
    lamb=3e-4, #0.01,
    beta=1e-7,
    clip=[0.01, 4])
 
rel_l1_error = RelativeL1Error(name="RelL1")
rel_l2_error = RelativeL2Error(name="RelL2")
dynamic_range = DynamicRange(name="DR")
dice_score = DiceScore(name="Dice", backCond=backCond)
measurement_error = MeasurementError(name="VoltageError", solver=solver_reco)

mesh_pos_reco = np.array(solver_reco.V_sigma.tabulate_dof_coordinates()[:, :2])

# generate phantom on coarse mesh
sigma_mesh = gen_conductivity(
    mesh_pos_reco[:, 0], mesh_pos_reco[:, 1], max_numInc=3, backCond=backCond
)
sigma_gt_coarse = Function(solver_reco.V_sigma)
sigma_gt_coarse.x.array[:] = sigma_mesh

# Interpolate conductivity on coarse mesh to dense mesh 
sigma_gt_fine = Function(solver_sim.V_sigma)
sigma_gt_fine.interpolate(
    sigma_gt_coarse,
    nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
        solver_sim.omega,
        solver_reco.V_sigma.element,
        solver_reco.omega,
        padding=1e-14,
    ),
)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
pred = np.array(sigma_gt_fine.x.array[:]).flatten()
im = ax1.tripcolor(
    tri_sim,
    pred,
    cmap="jet",
    shading="flat",
    vmin=0.01,
    vmax=3.0,
    edgecolor="k",
)
ax1.axis("image")
ax1.set_aspect("equal", adjustable="box")
ax1.set_title("Interplation to fine mesh")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.axis("off")

pred = np.array(sigma_gt_coarse.x.array[:]).flatten()
im = ax2.tripcolor(
    tri_reco,
    pred,
    cmap="jet",
    shading="flat",
    vmin=0.01,
    vmax=3.0,
    edgecolor="k",
)
ax2.axis("image")
ax2.set_aspect("equal", adjustable="box")
ax2.set_title("Coarse Mesh")
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
ax2.axis("off")

plt.show()


_, Umeas = solver_sim.forward_solve(sigma_gt_fine)
Umeas = np.array(Umeas)

_, Umeas_coarse = solver_reco.forward_solve(sigma_gt_coarse)
Umeas_coarse = np.array(Umeas_coarse)

# Add noise to measurements
U_noisy = Umeas + (delta1 * np.abs(Umeas) + delta2 * np.max(np.abs(Umeas))) * np.random.normal(
    size=Umeas.shape
)
# np.mean(np.abs(Umeas))


fig, axes = plt.subplots(4, 4, figsize=(14,14))

fig.suptitle("Measurements")
for idx, ax in enumerate(axes.ravel()):
    ax.set_title(f"Pattern {idx}")
    ax.plot(Umeas[idx, :], label="sim")
    ax.plot(U_noisy[idx,:], label="noisy")
    ax.plot(Umeas_coarse[idx,:], label="sim coarse")
    ax.legend()

plt.show()


var_meas = (delta1 * np.abs(U_noisy) + delta2 * np.max(np.abs(U_noisy))) ** 2
GammaInv = 1.0 / (np.maximum(var_meas.flatten(),1e-6))
GammaInv = torch.from_numpy(GammaInv).float().to(reconstructor.device)

print("var_meas: ", var_meas.min(), var_meas.max(), var_meas.shape)

print("GammaInv: ", GammaInv.min(), GammaInv.max(), GammaInv.shape)


reconstructor.GammaInv = GammaInv

sigma_init = Function(solver_reco.V_sigma)
sigma_init.x.array[:] = backCond

sigma_reco = reconstructor.forward(
    Umeas=Umeas, verbose=True, sigma_init=sigma_init
)

_, Usim = solver_reco.forward_solve(sigma_reco)
Usim = np.array(Umeas)

dr = dynamic_range(sigma_reco, sigma_gt_coarse)
dice = dice_score(sigma_reco, sigma_gt_coarse)
m_error = measurement_error(sigma_reco, Umeas)
l1_error = rel_l1_error(sigma_reco, sigma_gt_coarse)
l2_error = rel_l2_error(sigma_reco, sigma_gt_coarse)

print("Evaluation: ")
print("\t Dynamic Range: ", dr)
print("\t Measurement Error: ", m_error)
print("\t Relative L1-Error: ", l1_error)
print("\t Relative L2-Error: ", l2_error)
print("\t Dice Score: ", dice)

fig, axes = plt.subplots(4, 4, figsize=(14,14))

fig.suptitle("Measurements")
for idx, ax in enumerate(axes.ravel()):
    ax.set_title(f"Pattern {idx}")
    ax.plot(Usim[idx, :], label="re-simulated")
    ax.plot(U_noisy[idx,:], label="meas")
    ax.legend()

plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 6))

pred = np.array(sigma_gt_coarse.x.array[:]).flatten()
im = axes[0,0].tripcolor(
    tri_reco,
    pred,
    cmap="jet",
    shading="flat",
    vmin=0.01,
    vmax=3.0,
    edgecolor="k",
)
axes[0,0].axis("image")
axes[0,0].set_aspect("equal", adjustable="box")
axes[0,0].set_title("Ground truth")
fig.colorbar(im, ax=axes[0,0], fraction=0.046, pad=0.04)
axes[0,0].axis("off")

pred = np.array(sigma_reco.x.array[:]).flatten()
im = axes[0,1].tripcolor(
    tri_reco,
    pred,
    cmap="jet",
    shading="flat",
    vmin=0.01,
    vmax=3.0,
    edgecolor="k",
)
axes[0,1].axis("image")
axes[0,1].set_aspect("equal", adjustable="box")
axes[0,1].set_title("Reconstruction")
fig.colorbar(im, ax=axes[0,1], fraction=0.046, pad=0.04)
axes[0,1].axis("off")

pred = np.array(sigma_gt_coarse.x.array[:]).flatten()
im = axes[1,0].tripcolor(
    tri_reco,
    pred,
    cmap="jet",
    shading="flat",
    vmin=0.01,
    vmax=3.0,
)
axes[1,0].axis("image")
axes[1,0].set_aspect("equal", adjustable="box")
axes[1,0].set_title("Ground truth (without mesh)")
fig.colorbar(im, ax=axes[1,0], fraction=0.046, pad=0.04)
axes[1,0].axis("off")

pred = np.array(sigma_reco.x.array[:]).flatten()
im = axes[1,1].tripcolor(
    tri_reco,
    pred,
    cmap="jet",
    shading="flat",
    vmin=0.01,
    vmax=3.0,
)
axes[1,1].axis("image")
axes[1,1].set_aspect("equal", adjustable="box")
axes[1,1].set_title("Reconstruction (without mesh)")
fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
axes[1,1].axis("off")

plt.show()
