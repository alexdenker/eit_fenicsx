"""
min_{delta_s} 1/2 || F(s_background + delta_s) - U ||_2^2 + alpha || delta_s||_1


"""


import os 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation

from scipy.io import loadmat
from scipy.interpolate import interpn

from collections.abc import MutableMapping
import argparse
import wandb

import time 
import yaml 
from tqdm import tqdm 
from copy import deepcopy

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

import dolfinx
from dolfinx import io
from dolfinx import default_scalar_type
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem import (Constant, Function, functionspace, form, dirichletbc, 
                         locate_dofs_topological, assemble_scalar)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector

from petsc4py import PETSc
import ufl

from src import EIT



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

### get jacobian for sigma 
### get jacobian for sigma 
def compute_jacobian(eit_solver, sigma):

    u_list, U_list = eit_solver.forward_solve(sigma)

    dofs = eit_solver.dofs
    u_list = np.row_stack(u_list)

    ### Get matrix for lhs (same as in forward solve)
    petsc_mat = PETSc.Mat().createAIJ(size=eit_solver.M_complete.shape, 
                                    csr=(eit_solver.M_complete.indptr, 
                                        eit_solver.M_complete.indices, 
                                        eit_solver.M_complete.data))

    petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    petsc_mat.setValues([dofs+32], [dofs+32], [[0.0]], PETSc.InsertMode.INSERT_VALUES)
    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()
    petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

    lu_solver = PETSc.KSP().create(eit_solver.omega.comm)
    lu_solver.setOperators(petsc_mat)
    lu_solver.setType(PETSc.KSP.Type.PREONLY)
    lu_solver.getPC().setType(PETSc.PC.Type.LU)

    ## create empty PETSc vectors for RHS and solution u
    rhs_vec = PETSc.Vec()
    rhs_vec.create(PETSc.COMM_WORLD)
    rhs_vec.setSizes(dofs + 33)
    rhs_vec.setUp()

    sol_petsc = PETSc.Vec()
    sol_petsc.create(PETSc.COMM_WORLD)
    sol_petsc.setSizes(dofs + 33)
    sol_petsc.setUp()

    ### Now, we solve the problem for the jaciobian but where on the rhs we try 1 for each dof:
    ### save the 32 values at the electrodes -> linear combination of them lets us construct the jacobian
    sol_list = np.zeros((dofs, 32))

    start_time = time.time()
    for i in range(dofs):
        # Reset and set new value
        with rhs_vec.localForm() as loc_b:
            loc_b.set(0)

        rhs_vec.setValues([i], [1])
        rhs_vec.assemblyBegin()
        rhs_vec.assemblyEnd()

        # solve and save important values
        lu_solver.solve(rhs_vec, sol_petsc)
        sol_list[i, :] = sol_petsc[dofs:-1]

    print("solving helper problems took:", time.time() - start_time)

    ### Now we can construct the jacobian (right now only when sigma in DG 0)
    counter = 0

    num_of_cells = eit_solver.omega.topology.index_map(eit_solver.omega.topology.dim).size_local
    jacobian = np.zeros((u_list.shape[0], 32, num_of_cells))
    dofmap_V = eit_solver.V.dofmap
    dofmap_coords = eit_solver.V.tabulate_dof_coordinates()[:, :2]

    start_time = time.time()

    ## Iterate over all mesh cells = dof for DG 0 space 
    for cell in range(num_of_cells):
        # Get dof of u that belong to this cell and vertex coords:
        dof_of_cell = dofmap_V.cell_dofs(cell)
        vertex = dofmap_coords[dof_of_cell]

        # Build stiffness matrix for the element (only works for linear basis functions)
        # Check: https://en.wikipedia.org/wiki/Stiffness_matrix
        D = np.array([[vertex[2][0] - vertex[1][0], vertex[0][0] - vertex[2][0], vertex[1][0] - vertex[0][0]], 
                    [vertex[2][1] - vertex[1][1], vertex[0][1] - vertex[2][1], vertex[1][1] - vertex[0][1]]])

        edge_segements = zip(vertex, vertex[[1, 2, 0]])
        area = 0.5 * abs(sum(x0*y1 - x1*y0 for ((x0, y0), (x1, y1)) in edge_segements))

        local_stiffness_matrix = np.matmul(D.T, D) / (4.0 * area)

        # compute rhs 
        rhs_values = np.matmul(u_list[:, dof_of_cell], local_stiffness_matrix)
        
        # if cell <= 5:
        # #     print(dof_of_cell)
        # #     print(local_stiffness_matrix)
        #     print(rhs_values)

        # map rhs to correct dof and to the corresponding W solution of the helpers above
        jacobian[:, :, counter] = np.matmul(sol_list[dof_of_cell, :].T, rhs_values.T).T
        counter += 1

    
    return jacobian 



# load the reference data (use only the injection pattern of the challenge)
y_ref = loadmat('data/ref.mat') 
Injref = y_ref["Injref"]

x_img = loadmat(f'GroundTruths/true2.mat')["truth"]*1.0
sigma_img = np.zeros(x_img.shape)
sigma_img[x_img == 0.0] = 1.0
sigma_img[x_img == 1.0] = 0.04
sigma_img[x_img == 2.0] = 2.0

# We create the EIT solver with the matrix constructed using PETSc
z0 = 1e-6* np.ones(32) 
solver = EIT(Injref, z0, backend="PETSc")

# We use piecewise constant functions to approximate the solution
V_sigma = functionspace(solver.omega, ("DG", 0))

# We create the ground truth on the same mesh
sigma_gt_tmp = Function(V_sigma)
mesh_pos = np.array(V_sigma.tabulate_dof_coordinates()[:,:2])
sigma_gt_tmp.x.array[:] = image_to_mesh(np.flipud(sigma_img).T, mesh_pos)

sigma_gt = Function(V_sigma)
sigma_gt.interpolate(sigma_gt_tmp)

# simulate forward 
u_list, U = solver.forward_solve(sigma_gt)
U = np.asarray(U) 

Jacobian = solver.calc_jacobian(sigma_gt)

print(sigma_gt_tmp.x.array[:].shape)

print(Jacobian.shape)

Jacobian2 = compute_jacobian(solver, sigma_gt)
Jacobian2_reshape = Jacobian2.reshape(32*76, Jacobian2.shape[-1])
print(Jacobian2.shape)

print("Difference: ", np.linalg.norm(Jacobian2[0,:,:] - Jacobian2_reshape[0:32,:])/np.linalg.norm(Jacobian2[0,:,:]))

#print("Difference: ", np.linalg.norm(Jacobian2 - Jacobian)/np.linalg.norm(Jacobian2))


"""
print("Difference: ", np.linalg.norm(Jacobian2 - Jacobian)/np.linalg.norm(Jacobian))


print(Jacobian)

print(Jacobian2)

import matplotlib.pyplot as plt 

fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

ax1.matshow(Jacobian)

ax2.matshow(Jacobian2)

plt.show()
"""

#print(U)
#print(U.shape)
#print(Injref)
#print(Injref.shape)

#print(Injref[:,0])
#print(Injref[:,16])

#for i in range(20):
#    print(i,Injref[:, i] )


#print(U[0, :])
#print(U[16, :])


print("rank: ",np.linalg.matrix_rank(Injref.T))