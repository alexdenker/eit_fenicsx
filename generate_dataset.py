"""
Generation of the Dataset 

"""

import os 
import numpy as np
from scipy.io import loadmat

from tqdm import tqdm

from matplotlib.tri import Triangulation

from dolfinx.fem import Function, create_nonmatching_meshes_interpolation_data

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


from src import EIT, gen_conductivity, image_to_mesh, interpolate_mesh_to_mesh

part = "train_extra"
dataset_size = {"train_extra": 8000, "train": 2000, "val": 200, "test": 100}
start_seed = {"train": 0, "val": 5000, "test": 6000, "train_extra": 10000}

if part == "train_extra":
    save_part_name = "train"
else:
    save_part_name = part 
if not os.path.exists(f"dataset/{save_part_name}"):
    os.makedirs(f"dataset/{save_part_name}")

device = "cuda"

L = 16

data = loadmat("KIT4/data_mat_files/datamat_1_0.mat")

Injref = data["CurrentPattern"].T
# Injref = Injref[-15:, :]

z = 1e-6 * np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_dense.msh")

backCond = 1.31

mesh_pos1 = np.array(solver.V_sigma.tabulate_dof_coordinates()[:, :2])

solver_coarse = EIT(
    L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh"
)
mesh_pos = np.array(solver_coarse.V_sigma.tabulate_dof_coordinates()[:, :2])

np.save("dataset/injection_pattern.npy", Injref)

for i in tqdm(range(dataset_size[part])):
    np.random.seed(start_seed[part] + i)

    # generate phantom on coarse mesh
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=backCond
    )
    sigma_gt_coarse = Function(solver_coarse.V_sigma)
    sigma_gt_coarse.x.array[:] = sigma_mesh

    sigma_gt_fine = Function(solver.V_sigma)
    sigma_gt_fine.interpolate(
        sigma_gt_coarse,
        nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
            solver.omega,
            solver_coarse.V_sigma.element,
            solver_coarse.omega,
            padding=1e-14,
        ),
    )

    _, Umeas = solver.forward_solve(sigma_gt_fine)
    Umeas = np.array(Umeas)

    xy = solver.omega.geometry.x
    cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))

    tri = Triangulation(xy[:, 0], xy[:, 1], cells)

    xy = solver_coarse.omega.geometry.x
    cells = solver_coarse.omega.geometry.dofmap.reshape(
        (-1, solver_coarse.omega.topology.dim + 1)
    )
    tri_coarse = Triangulation(xy[:, 0], xy[:, 1], cells)

    np.save("dataset/mesh_points.npy", np.array(xy))
    np.save("dataset/cells.npy", np.array(cells))

    if part == "train_extra":
        save_idx = dataset_size["train"] + i 
    else:
        save_idx = i 
    np.save(f"dataset/{save_part_name}/sigma_{save_idx}.npy", sigma_gt_coarse.x.array[:].flatten())
    np.save(f"dataset/{save_part_name}/Umeas_{save_idx}.npy", Umeas)


    if part == "test":
        delta = 0.005
        U_noisy = Umeas + delta * np.mean(np.abs(Umeas)) * np.random.normal(
            size=Umeas.shape
        )
        np.save(f"dataset/{save_part_name}/Umeas_noisy_{save_idx}.npy", U_noisy)

        # print(U_noisy.shape)
        # plt.figure()
        # plt.plot(Umeas[0,:], label="clean")
        # plt.plot(U_noisy[0,:], label="noisy")
        # plt.legend()
        # plt.show()

    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
    im = ax1.tripcolor(tri, sigma_gt_fine.x.array[:].flatten(), cmap='jet', shading='flat', vmin=0.01, vmax=4.0,edgecolors='k')
    ax1.axis('image')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title("Phantom")
    ax1.axis("off")
    fig.colorbar(im, ax=ax1,fraction=0.046, pad=0.04)

    im = ax2.tripcolor(tri_coarse, sigma_gt_coarse.x.array[:].flatten(), cmap='jet', shading='flat', vmin=0.01, vmax=4.0,edgecolors='k')
    ax2.axis('image')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title("Phantom (coarse)")
    ax2.axis("off")
    fig.colorbar(im, ax=ax2,fraction=0.046, pad=0.04)

    plt.show()
    """
