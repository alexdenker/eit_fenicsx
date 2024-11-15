"""
Implementation of Gauss-Newton 

"""

import numpy as np 
from scipy.io import loadmat

import torch 

from matplotlib.tri import Triangulation

from dolfinx.fem import Function, functionspace

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation


from src.eit_forward_fenicsx import EIT
from src.gauss_newton import GaussNewtonSolver

device = "cuda"

L = 32

y_ref = loadmat('data/ref.mat') 
Injref = y_ref["Injref"].T

z = 1e-6*np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KTC2023_mesh.msh")

# We use piecewise constant functions to approximate the solution
V_sigma = functionspace(solver.omega, ("DG", 0))

sigma_background = Function(solver.V) 
sigma_background.interpolate(lambda x: 0.8*np.ones_like(x[0]))

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

Lprior = np.load("L_KTC.npy")

R = Lprior.T @ Lprior
R = torch.from_numpy(R).float().to(device)


y = loadmat(f'data/data2.mat')
Uel = y["Uel"].reshape(76, 31)

Uel_background = y_ref["Uelref"].reshape(76, 31)


B = y_ref["Mpat"].T
Bf = np.vstack([B, np.ones(B.shape[-1])])

# collect measurements of captial U = B^{-1} Uel
U = [] 
U_background = [] 
for i in range(Uel.shape[0]):
    U.append(np.linalg.solve(Bf, np.hstack([Uel[i,:], np.array([0])])))
    U_background.append(np.linalg.solve(Bf, np.hstack([Uel_background[i,:], np.array([0])])))

Umeas = np.stack(U).flatten()
Umeas_background = np.stack(U_background).flatten()

noise_percentage = 0.05
noise_percentage2 = 0.025
var_meas = np.power(((noise_percentage) * (np.abs(Umeas))),2)
var_meas = var_meas + np.power((noise_percentage2) * np.max(np.abs(Umeas)),2)
GammaInv = 1./(var_meas + 5e-2)
GammaInv = torch.from_numpy(GammaInv).float().to(device)


sigma_init = Function(solver.V_sigma)
sigma_init.x.array[:] = 0.8

gauss_newton_solver = GaussNewtonSolver(solver, device=device)

sigma = gauss_newton_solver.reconstruct(Umeas=Umeas,
                                        sigma_init=sigma_init,
                                        num_steps=20,
                                        R=1e-2*R,
                                        GammaInv=GammaInv,
                                        clip=[0.001, 5.0],
                                        verbose=True)


fig, ax1 = plt.subplots(1,1, figsize=(9,9))
im = ax1.tripcolor(tri, sigma.flatten(), cmap='jet', shading='flat')
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title("Prediction")
fig.colorbar(im, ax=ax1)

plt.show()
