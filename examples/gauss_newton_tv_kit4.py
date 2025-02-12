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

This example requires to download the KIT4 dataset: https://zenodo.org/records/1203914

"""


import numpy as np
import os
import time
import sys

import yaml
import matplotlib
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.io import loadmat

import torch
from omegaconf import OmegaConf

from dolfinx.fem import Function, create_nonmatching_meshes_interpolation_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eit_forward_fenicsx import EIT
from src.gauss_newton import GaussNewtonSolverTV
from src.performance_metrics import (
    RelativeL1Error,
    DiceScore,
    DynamicRange,
    MeasurementError,
    RelativeL2Error
)

import time

delta1 = 0.1  # noise level
delta2 = 0.001 
L = 16 # number of electrodes 
backCond = 1.31 # background conductivity 
z = np.array(
            [
                0.00880276,
                0.00938687,
                0.00989395,
                0.01039582,
                0.00948009,
                0.00943006,
                0.01016697,
                0.0088116,
                0.00802456,
                0.0090383,
                0.00907472,
                0.00847228,
                0.00814984,
                0.00877861,
                0.00841414,
                0.00877331,
            ]
        ) # (estimated) contact impedance 

test_data = "3_2"
data = loadmat(f"KIT4/data_mat_files/datamat_{test_data}.mat")
Injref = data["CurrentPattern"].T


im_frame = Image.open(f"KIT4/target_photos/fantom_{test_data}.jpg")
np_frame = np.array(im_frame) / 255.0
target_image = torch.from_numpy(np_frame).float()

# we also load the data of the empty watertank, this is used for delta U in the first step of Gauss-Newton
data_watertank = loadmat("KIT4/data_mat_files/datamat_1_0.mat")
Uel_background = data_watertank["Uel"].T


B = data["MeasPattern"].T

Injref = data["CurrentPattern"].T
Uel = data["Uel"].T

Bf = np.vstack([B, np.ones(B.shape[-1])])

U = []
U_background = []
for i in range(Uel.shape[0]):
    exU = np.hstack([Uel[i, :], np.array([0])])
    U_sol, res, _, _ = np.linalg.lstsq(Bf, np.hstack([Uel[i, :], np.array([0])]))
    U.append(U_sol)

    U_sol, res, _, _ = np.linalg.lstsq(
        Bf, np.hstack([Uel_background[i, :], np.array([0])])
    )
    U_background.append(U_sol)


Uel = np.stack(U)
Uel_background = np.stack(U_background)

print(Uel.shape, Uel_background.shape)

# We set up two instances of the EIT forward operator: a coarse mesh and a dense mesh
# We use the dense mesh to simulate the measurements and the coarse mesh for reconstruction
solver_reco = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")


xy = solver_reco.omega.geometry.x
cells = solver_reco.omega.geometry.dofmap.reshape((-1, solver_reco.omega.topology.dim + 1))
tri_reco = Triangulation(xy[:, 0], xy[:, 1], cells)

reconstructor = GaussNewtonSolverTV(
    eit_solver=solver_reco,
    Uel_background=Uel_background,
    device="cuda",
    num_steps=8,
    lamb=5e-3, #0.01,
    beta=1e-7,
    clip=[0.01, 4])
 
rel_l1_error = RelativeL1Error(name="RelL1")
rel_l2_error = RelativeL2Error(name="RelL2")
dynamic_range = DynamicRange(name="DR")
dice_score = DiceScore(name="Dice", backCond=backCond)
measurement_error = MeasurementError(name="VoltageError", solver=solver_reco)

mesh_pos_reco = np.array(solver_reco.V_sigma.tabulate_dof_coordinates()[:, :2])





var_meas = (delta1 * np.abs(U_background) + delta2 * np.max(np.abs(U_background))) ** 2
GammaInv = 1.0 / (np.maximum(var_meas.flatten(),1e-5))
GammaInv = torch.from_numpy(GammaInv).float().to(reconstructor.device)

print("var_meas: ", var_meas.min(), var_meas.max(), var_meas.shape)

print("GammaInv: ", GammaInv.min(), GammaInv.max(), GammaInv.shape)


reconstructor.GammaInv = GammaInv

sigma_init = Function(solver_reco.V_sigma)
sigma_init.x.array[:] = backCond

sigma_reco = reconstructor.forward(
    Umeas=Uel, verbose=True, sigma_init=sigma_init
)

_, Usim = solver_reco.forward_solve(sigma_reco)
Usim = np.array(Usim)

m_error = measurement_error(sigma_reco, Uel)

print("Evaluation: ")
print("\t Measurement Error: ", m_error)

fig, axes = plt.subplots(4, 4, figsize=(14,14))

fig.suptitle("Measurements")
for idx, ax in enumerate(axes.ravel()):
    ax.set_title(f"Pattern {idx}")
    ax.plot(Usim[idx, :], label="re-simulated")
    ax.plot(Uel[idx,:], label="meas")
    ax.plot(Uel_background[idx,:], label="background")

    ax.legend()

plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 6))

im = axes[0,0].imshow(target_image)

axes[0,0].axis("image")
axes[0,0].set_aspect("equal", adjustable="box")
axes[0,0].set_title("Ground truth")
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
