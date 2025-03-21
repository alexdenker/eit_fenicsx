import numpy as np


import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import time
from tqdm import tqdm
import torch
from scipy.sparse import csr_array


from dolfinx.fem import Function


from src import EIT, gen_conductivity, CEMModule, current_method

def construct_tv_matrix(omega):
    omega.topology.create_connectivity(1, 2)  # Facet-to-cell connectivity
    omega.topology.create_connectivity(2, 1)  # Cell-to-facet connectivity

    # Number of cells in the mesh
    num_cells = omega.topology.index_map(2).size_local

    cell_to_edge = omega.topology.connectivity(2, 1)
    # cell_to_edge connects a cell to its border, every cell has always three borders because the cell is a triangle
    # e.g. 0: [4 1 0] "Cell 0 connects to border 4, 1 and 0"

    edge_to_cell = omega.topology.connectivity(1, 2)
    # edge_to_cell connects every border to a cell, here each border always has 1 or 2 cells
    # e.g. 7011: [4593 4600 ] "Border 7011 is part of cell 4593 and 4600" => This means that cells 4593 and 4600 are connected

    rows = []
    cols = []
    data = []

    row_idx = 0
    for cell in range(num_cells):
        # find borders of cell
        adjacent_edges = cell_to_edge.links(cell)
        for edge in adjacent_edges:
            # find cells connected to this border
            adjacent_cells = edge_to_cell.links(edge)
            if (
                len(adjacent_cells) > 1
            ):  # only look at the parts where we have two cells
                rows.append(row_idx)
                rows.append(row_idx)
                cols.append(adjacent_cells[0])
                cols.append(adjacent_cells[1])
                data.append(1)
                data.append(-1)

                row_idx += 1

    Lcsr = csr_array((data, (rows, cols)), shape=(row_idx, num_cells))

    Lcoo = Lcsr.tocoo()

    values = Lcoo.data
    indices = np.vstack((Lcoo.row, Lcoo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = Lcoo.shape

    #print("SHAPE:", shape)
    Ltorch_tv = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    # L = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return Ltorch_tv


device = "cuda"

optim = "adam" #"adam"  # "lbfgs"

L = 16
backCond = 1.0

Injref = np.concatenate([current_method(L=L, l=L//2, method=1,value=1.5), current_method(L=L, l=L, method=2,value=1.5)])
#Injref = current_method(L=L, l=L - 1, method=5, value=1.5)

z = 1e-6 * np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

mesh_pos = np.array(solver.V_sigma.tabulate_dof_coordinates()[:, :2])

np.random.seed(16)  # 14: works well
sigma_mesh = gen_conductivity(
    mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=backCond
)
sigma_gt_vsigma = Function(solver.V_sigma)
sigma_gt_vsigma.x.array[:] = sigma_mesh

sigma_gt = Function(solver.V)
sigma_gt.interpolate(sigma_gt_vsigma)

sigma_background = Function(solver.V_sigma)
sigma_background.interpolate(lambda x: backCond * np.ones_like(x[0]))

sigma_background_torch = (
    torch.from_numpy(sigma_background.x.array[:]).float().to(device).unsqueeze(0)
)

Ltorch_tv = construct_tv_matrix(solver.omega)


# We simulate the measurements using our forward solver
_, U = solver.forward_solve(sigma_gt)
Umeas = np.array(U)

noise_percentage = 0.01
var_meas = (noise_percentage * np.abs(Umeas)) ** 2
# Umeas = Umeas + np.sqrt(var_meas) * np.random.normal(size=Umeas.shape)

eit_module = CEMModule(
    eit_solver=solver, mode="jacobian", kappa=0.005, gradient_smooting=True
)  # kappa=0.028

sigma_torch = torch.nn.Parameter(torch.zeros_like(sigma_background_torch))

print(sigma_torch.shape, Ltorch_tv.shape)
Ltorch_tv = Ltorch_tv.to(sigma_torch.device)

if optim == "adam":
    optimizer = torch.optim.Adam([sigma_torch], lr=0.05)

elif optim == "lbfgs":
    optimizer = torch.optim.LBFGS([sigma_torch], lr=0.2, max_iter=12)
else:
    raise NotImplementedError
Umeas_torch = torch.from_numpy(Umeas).unsqueeze(0).float().to(device)

print("Number of parameters: ", sigma_torch.shape.numel())

num_iterations = 50 
tol = 1e-4

alpha_l1 = 0.0 #0.002  # 0.01 #0.04 #0.008 #0.015
alpha_tv = 12.0
for i in tqdm(range(num_iterations)):
    sigma_old = Function(solver.V_sigma)
    sigma_old.x.array[:] = (
        (sigma_background_torch + sigma_torch).detach().cpu().numpy()[0, :]
    )

    if optim == "adam":
        optimizer.zero_grad()
        Upred = eit_module(sigma_background_torch + sigma_torch)

        loss_mse = torch.sum((Umeas_torch - Upred) ** 2)
        loss_l1 = torch.sum(torch.abs(sigma_torch))
        loss_tv = torch.mean(torch.abs(torch.matmul(Ltorch_tv, sigma_torch.T)))
        loss = loss_mse + alpha_l1 * loss_l1 + alpha_tv * loss_tv
        print("Loss: ", loss_mse.item(), loss_l1.item(), loss_tv.item())
        loss.backward()
        optimizer.step()
    elif optim == "lbfgs":

        def closure():
            optimizer.zero_grad()

            sigma_torch.data.clamp_(-backCond + 0.01, 3.0)

            Upred = eit_module(sigma_background_torch + sigma_torch)

            loss_mse = torch.sum((Umeas_torch - Upred) ** 2)
            loss_l1 = torch.sum(torch.abs(sigma_torch))

            loss = loss_mse + alpha_l1 * loss_l1
            loss.backward()
            return loss

        optimizer.step(closure)
    else:
        raise NotImplementedError

    sigma_torch.data.clamp_(-backCond + 0.01, 3.0)

    sigma_j = Function(solver.V_sigma)
    sigma_j.x.array[:] = (
        (sigma_background_torch + sigma_torch).detach().cpu().numpy()[0, :]
    )

    sigma_grad = Function(solver.V_sigma)
    sigma_grad.x.array[:] = (
        sigma_torch.grad.detach().cpu().numpy()[0, :]
    )


    rel_change = np.linalg.norm(sigma_j.x.array[:] - sigma_old.x.array[:])/ np.linalg.norm(sigma_j.x.array[:])
    print("Relative Change: ", rel_change)

    if rel_change < tol:
        break

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

    im = ax1.tripcolor(
        tri,
        np.array(sigma_j.x.array[:]).flatten(),
        cmap="jet",
        shading="flat",
        vmin=0.01,
        vmax=2,
    )
    ax1.set_title("Reconstruction")
    ax1.axis("image")
    ax1.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax1)
    ax1.axis("off")

    im = ax2.tripcolor(
        tri,
        np.array(sigma_gt_vsigma.x.array[:]).flatten(),
        cmap="jet",
        shading="flat",
        vmin=0.01,
        vmax=2,
    )
    ax2.set_title("Ground truth")
    ax2.axis("image")
    ax2.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax2)
    ax2.axis("off")

    im = ax3.tripcolor(
        tri,
        np.array(sigma_grad.x.array[:]).flatten(),
        cmap="jet",
        shading="flat",
    )
    ax3.set_title("Gradient")
    ax3.axis("image")
    ax3.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=ax3)
    ax3.axis("off")

    plt.show()
