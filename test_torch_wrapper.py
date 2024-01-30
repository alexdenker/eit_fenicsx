
import os 
import numpy as np 

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt 
from matplotlib import colors
from matplotlib.tri import Triangulation
from scipy.io import loadmat
import time 
from tqdm import tqdm 
import torch 
from scipy.sparse import csr_array

from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.interpolate import interpn
from matplotlib.tri import Triangulation

from dolfinx.fem import Function, FunctionSpace
from dolfinx import io

from src import EIT, CEMModule

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




def image_to_mesh(x, mesh_pos):

    #sigma = np.ones((Meshsim.g.shape[0], 1))

    pixwidth = 0.23 / 256
    # pixcenter_x = np.arange(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, pixwidth)
    pixcenter_x = pixcenter_y = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")
    pixcenters = np.column_stack((X.ravel(), Y.ravel()))

    sigma = interpn([pixcenter_x, pixcenter_y], x, mesh_pos, 
        bounds_error=False, fill_value=1.0, method="nearest")

    return sigma

y_ref = loadmat('data/ref.mat') #load the reference data
Injref = y_ref["Injref"]


sigma_img = np.ones((256, 256))
pixwidth = 0.23 / 256

pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
pixcenter_y = pixcenter_x
X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
sigma_img[((X + 0.04)**2 + Y**2) < 0.001] = 4.0



plt.figure()
plt.imshow(sigma_img)
plt.colorbar()
plt.savefig("results/ground_truth.png")
plt.close()


z0 = 1e-6* np.ones(32) #1./y0

solver = EIT(Injref, z0)

V_sigma = FunctionSpace(solver.omega, ("DG", 0))

def sigma_function(x):
    return np.ones_like(x[0])

mesh_pos = np.array(V_sigma.tabulate_dof_coordinates()[:,:2])

sigma_gt = Function(V_sigma)


sigma_gt.x.array[:] = image_to_mesh(sigma_img, mesh_pos)

eit_module = CEMModule(solver, kappa=0.05, gradient_smooting=True)

_, Umeas = solver.forward_solve(sigma_gt)
Umeas = np.array(Umeas)

mesh_neighbors = np.load("data/mesh_neighbour_matrix.npy")

Ltv = create_Ltv(mesh_neighbors)
#print(image_to_mesh(sigma_img, mesh_pos).shape)

tri = Triangulation(mesh_pos[:, 0], mesh_pos[:, 1])


sigma0 = 1.0
sigma_background = torch.ones((1, mesh_pos.shape[0]))*sigma0
sigma_torch = torch.nn.Parameter(torch.zeros_like(sigma_background))

optimizer = torch.optim.LBFGS([sigma_torch], lr=0.5, max_iter=10)#2e-6)
Umeas_torch = torch.from_numpy(Umeas).unsqueeze(0).float()
import time 

alpha_tv = 5.0
alpha_l1 = 0.03
for i in tqdm(range(200)):
    time1 = time.time() 
    def closure():
        optimizer.zero_grad()
        #print(torch.min(sigma_torch), torch.max(sigma_torch))
        sigma_torch.data.clamp_(-0.74, 4.)

        Upred = eit_module(sigma_background + sigma_torch)

        loss_mse = torch.sum((Umeas_torch - Upred)**2) 
        loss_tv = torch.sum(torch.abs(torch.matmul(Ltv, sigma_torch.T))) 
        loss_l1 = torch.sum(torch.abs(sigma_torch))
        #print(loss_mse.item(), alpha_l1*loss_l1.item(),alpha_tv*loss_tv.item())
        loss = loss_mse + alpha_l1 * loss_l1 + alpha_tv * loss_tv 
        loss.backward() #+ self.alpha*torch.sum(torch.abs(self.Ltorch @ deltaSigma))
        #print(torch.min(sigma_torch.grad), torch.max(sigma_torch.grad))
        return loss
    
    #sigma_grad = sigma_torch.grad.detach().cpu().numpy()
    #print(sigma_grad.shape)
    
    optimizer.step(closure)

    sigma_torch.data.clamp_(-0.74, 4.)
    time2 = time.time()
    
    print("ONE GRADIENT STEP TAKES: ", time2-time1, "s")
    
    sigma_j = Function(V_sigma)
    sigma_j.x.array[:] = (sigma_background + sigma_torch).detach().cpu().numpy()[0,:]

    fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(16,6))

    im = ax0.tripcolor(tri, np.array(sigma_j.x.array[:]).flatten(), cmap='jet')
    ax0.set_title("Reconstruction (own color range)")
    ax0.axis('image')
    ax0.set_aspect('equal', adjustable='box')
    fig.colorbar(im, ax=ax0)

    im = ax1.tripcolor(tri, np.array(sigma_j.x.array[:]).flatten(), cmap='jet', vmin=np.min(np.array(sigma_gt.x.array[:])), vmax=2.5)
    ax1.set_title("Reconstruction (same color range)")
    ax1.axis('image')
    ax1.set_aspect('equal', adjustable='box')
    fig.colorbar(im, ax=ax1)

    im = ax2.tripcolor(tri, np.array(sigma_gt.x.array[:]).flatten(), cmap='jet')
    ax2.set_title("Ground truth")
    ax2.axis('image')
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(im, ax=ax2)
    plt.savefig(f"results/test_{i}.png")
    plt.close()
