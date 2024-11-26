
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

import torch 

from dolfinx.fem import Function, assemble_scalar, form
import ufl 

from src.eit_forward_fenicsx import EIT
from src.random_ellipses import gen_conductivity
from src.sparsity_reconstruction import L1Sparsity
from src.utils import current_method
from src.regulariser import create_smoothness_regulariser
from src.gauss_newton import GaussNewtonSolver

def compute_relative_l1_error(sigma_rec, sigma_gt):

    diff = abs(sigma_rec - sigma_gt) * ufl.dx 
    diff = assemble_scalar(form(diff))

    norm = abs(sigma_gt) * ufl.dx
    norm = assemble_scalar(form(norm))
    
    return diff/norm 



device = "cuda"

L = 16
backCond = 1.0

Injref = np.concatenate([current_method(L=L, l=L//2, method=1,value=1.5), current_method(L=L, l=L, method=2,value=1.5)])

z = 1e-6*np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(solver.V_sigma.tabulate_dof_coordinates()[:,:2])

np.random.seed(16) # 14: works well
sigma_mesh = gen_conductivity(mesh_pos[:,0], mesh_pos[:,1], max_numInc=3, backCond=backCond)
sigma_gt_vsigma = Function(solver.V_sigma)
sigma_gt_vsigma.x.array[:] = sigma_mesh

sigma_gt = Function(solver.V)
sigma_gt.interpolate(sigma_gt_vsigma)

# We simulate the measurements using our forward solver
_, U = solver.forward_solve(sigma_gt)
Umeas = np.array(U)

noise_percentage = 0.01
var_meas = (noise_percentage * np.abs(Umeas))**2
Umeas = Umeas + np.sqrt(var_meas) * np.random.normal(size=Umeas.shape)

### Hyperparameters for L1-Reg
l1 = 0.01 # smallest possible conductivity 
l2 = 4.0 # largest conductivity 

alpha = 0.001 #0.0001
kappa = 0.03

l1_reconstructor = L1Sparsity(eit_solver=solver,
                            backCond=backCond,
                            kappa=kappa,
                            clip=[l1,l2],
                            max_iter=200,
                            stopping_criterion=5e-4,
                            step_min=1e-6,
                            initial_step_size=0.2)

sigma_reco_l1, _ = l1_reconstructor.reconstruct(Umeas=Umeas, alpha=alpha)

sigma_reco_l1_vsigma = Function(solver.V_sigma)
sigma_reco_l1_vsigma.interpolate(sigma_reco_l1)

try:
    Lprior = np.load("data/L_KIT4_mesh_coarse.npy")
except FileNotFoundError:
    print("Regulariser matrix was not found. Rebuild it...this may take a while")

    create_smoothness_regulariser(solver.omega, "data/L_KIT4_mesh_coarse.npy", corrlength=0.2, std=0.15)
    Lprior = np.load("data/L_KIT4_mesh_coarse.npy")

Lprior = torch.from_numpy(Lprior).float().to(device)

R = Lprior.T @ Lprior

Umeas_flatten = np.array(Umeas).flatten()

GammaInv = 1./(var_meas.flatten() + 0.001)
GammaInv = torch.from_numpy(GammaInv).float().to(device)

sigma_init = Function(solver.V_sigma)
sigma_init.x.array[:] = backCond

gauss_newton_solver = GaussNewtonSolver(solver, device=device)

sigma = gauss_newton_solver.reconstruct(Umeas=Umeas_flatten,
                                        sigma_init=sigma_init,
                                        num_steps=10,
                                        R=R, 
                                        lamb=0.6, #8e-4,
                                        GammaInv=GammaInv,
                                        clip=[0.001, 3.0],
                                        verbose=True)


sigma_rec = Function(solver.V_sigma)
sigma_rec.x.array[:] = sigma 

rel_error_l1 = compute_relative_l1_error(sigma_reco_l1_vsigma, sigma_gt_vsigma)
rel_error_gn = compute_relative_l1_error(sigma_rec, sigma_gt_vsigma)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(19,6))

pred = np.array(sigma_reco_l1_vsigma.x.array[:]).flatten()
im = ax1.tripcolor(tri, pred, cmap='jet', shading='flat',vmin=0.01, vmax=2.0)
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
ax1.set_title(f"L1-Sparsity, relative L1 error={np.format_float_positional(rel_error_l1,4)}")
fig.colorbar(im, ax=ax1)

pred = np.array(sigma_gt_vsigma.x.array[:]).flatten()
im = ax2.tripcolor(tri, pred, cmap='jet', shading='flat',vmin=0.01, vmax=2.0)
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
ax2.set_title("GT")
fig.colorbar(im, ax=ax2)

pred = np.array(sigma_rec.x.array[:]).flatten()
im = ax3.tripcolor(tri, pred, cmap='jet', shading='flat', vmin=0.01, vmax=2.0)
ax3.axis('image')
ax3.set_aspect('equal', adjustable='box')
ax3.set_title(f"Gauss-Newton, relative L1 error={np.format_float_positional(rel_error_gn,4)}")
fig.colorbar(im, ax=ax3)

plt.savefig("example_reconstruction.png", bbox_inches='tight')

plt.show()
