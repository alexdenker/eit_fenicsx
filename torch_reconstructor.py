
import numpy as np 


import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation
import time 
from tqdm import tqdm 
import torch 
from scipy.sparse import csr_array


from dolfinx.fem import Function


from src.eit_forward_fenicsx import EIT
from src.torch_wrapper import CEMModule
from src.random_ellipses import gen_conductivity
from src.utils import current_method

def create_Ltv(mesh_neighbors):
    data = [] 
    row = [] 
    column = [] 

    saved_edges = []
    row_id = 0
    for k in range(mesh_neighbors.shape[0]):
        edges = np.where(mesh_neighbors[k,:] > 0)[0]
        
        for e in edges:
            #if not [min(k, e), max(k, e)] in saved_edges:
            data.append(1.)
            data.append(-1.)
                
            row.append(row_id)
            row.append(row_id)

            column.append(k)
            column.append(e)

            saved_edges.append([min(k, e), max(k, e)])
                
            row_id += 1

    Lcsr = csr_array((np.array(data), (np.array(row), np.array(column))))
    values = torch.from_numpy(np.array(data)).float()

    Lcoo = Lcsr.tocoo()

        
    values = Lcoo.data
    indices = np.vstack((Lcoo.row, Lcoo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = Lcoo.shape

    print("SHAPE:", shape)
    L = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    #L = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return L  

device = "cuda"

optim = "adam" # "lbfgs"

L = 16
backCond = 1.0

#Injref = np.concatenate([current_method(L=L, l=L//2, method=1,value=1.5), current_method(L=L, l=L, method=2,value=1.5)])
Injref = current_method(L=L, l=L-1, method=5,value=1.5)

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

sigma_background = Function(solver.V) 
sigma_background.interpolate(lambda x: backCond*np.ones_like(x[0]))

sigma_background_torch = torch.from_numpy(sigma_background.x.array[:]).float().to(device).unsqueeze(0)


# We simulate the measurements using our forward solver
_, U = solver.forward_solve(sigma_gt)
Umeas = np.array(U)

noise_percentage = 0.01
var_meas = (noise_percentage * np.abs(Umeas))**2
Umeas = Umeas + np.sqrt(var_meas) * np.random.normal(size=Umeas.shape)

eit_module = CEMModule(eit_solver=solver, kappa=0.028, gradient_smooting=True)

sigma_torch = torch.nn.Parameter(torch.zeros_like(sigma_background_torch))

if optim == "adam":
    optimizer = torch.optim.Adam([sigma_torch], lr=0.1)

elif optim == "lbfgs":
    optimizer = torch.optim.LBFGS([sigma_torch], lr=0.2, max_iter=12)
else:
    raise NotImplementedError
Umeas_torch = torch.from_numpy(Umeas).unsqueeze(0).float().to(device)

print("Number of parameters: ", sigma_torch.shape.numel())


alpha_l1 = 0.1 #0.008 #0.015
for i in tqdm(range(80)):
    sigma_old = Function(solver.V)
    sigma_old.x.array[:] = (sigma_background_torch + sigma_torch).detach().cpu().numpy()[0,:]

    if optim == "adam":
        optimizer.zero_grad()
        Upred = eit_module(sigma_background_torch + sigma_torch)

        loss_mse = torch.sum((Umeas_torch - Upred)**2) 
        loss_l1 = torch.sum(torch.abs(sigma_torch))

        loss = loss_mse + alpha_l1 * loss_l1 
        loss.backward() 
        optimizer.step() 
    elif optim == "lbfgs":
            
        def closure():
            optimizer.zero_grad()

            sigma_torch.data.clamp_(-backCond + 0.01, 3.)

            Upred = eit_module(sigma_background_torch + sigma_torch)

            loss_mse = torch.sum((Umeas_torch - Upred)**2) 
            loss_l1 = torch.sum(torch.abs(sigma_torch))

            loss = loss_mse + alpha_l1 * loss_l1 
            loss.backward() 
            return loss
        
        optimizer.step(closure)
    else:
        raise NotImplementedError
    
    sigma_torch.data.clamp_(-backCond + 0.01, 3.)
    
    sigma_j = Function(solver.V)
    sigma_j.x.array[:] = (sigma_background_torch + sigma_torch).detach().cpu().numpy()[0,:]

    print("Relative Change: ", np.linalg.norm(sigma_j.x.array[:] - sigma_old.x.array[:])/np.linalg.norm(sigma_j.x.array[:]))

fig, (ax2, ax1) = plt.subplots(1,2,figsize=(16,6))

im = ax1.tripcolor(tri, np.array(sigma_j.x.array[:]).flatten(), cmap='jet', shading='gouraud', vmin=0.01, vmax=2)
ax1.set_title("Reconstruction")
ax1.axis('image')
ax1.set_aspect('equal', adjustable='box')
fig.colorbar(im, ax=ax1)
ax1.axis("off")

im = ax2.tripcolor(tri, np.array(sigma_gt.x.array[:]).flatten(), cmap='jet', shading='gouraud', vmin=0.01, vmax=2)
ax2.set_title("Ground truth")
ax2.axis('image')
ax2.set_aspect('equal', adjustable='box')
fig.colorbar(im, ax=ax2)
ax2.axis("off")
plt.savefig("imgs/torch_reconstruction.png", bbox_inches="tight")
plt.show()
