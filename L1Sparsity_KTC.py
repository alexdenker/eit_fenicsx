"""
min_{delta_s} 1/2 || F(s_background + delta_s) - U ||_2^2 + alpha || delta_s||_1

L1-Sparsity, see: https://www.sciencedirect.com/science/article/pii/S0377042711005140

"""


import os 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.tri import Triangulation
from scipy.io import loadmat

import time 

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

import dolfinx
from dolfinx import default_scalar_type
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem import (Constant, Function, functionspace, form, dirichletbc, 
                         locate_dofs_topological, assemble_scalar)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector

from petsc4py import PETSc
import ufl


from src.eit_forward_fenicsx import EIT

L = 32

y_ref = loadmat('data/ref.mat') 
Injref = y_ref["Injref"].T

z = 1e-6*np.ones(L)
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KTC2023_mesh.msh")

xy = solver.omega.geometry.x
cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
tri = Triangulation(xy[:, 0], xy[:, 1], cells)

y = loadmat(f'data/data3.mat')
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

Umeas = np.stack(U)
Umeas_background = np.stack(U_background)

noise_percentage = 0.05
noise_percentage2 = 0.025
var_meas = np.power(((noise_percentage) * (np.abs(Umeas))),2)
var_meas = var_meas + np.power((noise_percentage2) * np.max(np.abs(Umeas)),2)
GammaInv = 1./(var_meas + 5e-2)



# We need the the surface area of all mesh elements 
# to correctly implement the sparsity term
v = ufl.TestFunction(solver.V_sigma)
cell_area_form = dolfinx.fem.form(v*ufl.dx)
cell_area = dolfinx.fem.assemble_vector(cell_area_form)
cell_area = np.array(cell_area.array)



### Hyperparameters for L1-Reg
l1 = 0.01 # smallest possible conductivity 
l2 = 4.0 # largest conductivity 
sigma0 = 0.8 # background conductivity 
alpha = 4.0# L1 regularisation strength
kappa = 0.005

### set up gradient smoothing ###

# smooting level
kappa = Constant(solver.omega, default_scalar_type(kappa))

### Create boundary condition (zero boundary)
fdim = solver.omega.topology.dim - 1
boundary_facets = exterior_facet_indices(solver.omega.topology)
bc = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(solver.V, fdim, boundary_facets), solver.V)

# create the LHS matrix 
# solver.u is in FunctionSpace(self.omega, ("Lagrange", 1))
# piecewise linear functions
a = (ufl.inner(kappa * ufl.grad(solver.u), ufl.grad(solver.phi)) + solver.u * solver.phi) * ufl.dx
bilinear_form = form(a)
A_smooth = assemble_matrix(bilinear_form, bcs=[bc])
A_smooth.assemble()

class GradientSmoother():
    def __init__(self, A_smooth, dofs, backend="Scipy"):

        if backend == "Scipy":
 
            ai, aj, av = A_smooth.getValuesCSR()
            scipy_A = csr_matrix((av, aj, ai))
            scipy_A.resize(dofs, dofs)

            self.smooting_solver = factorized(scipy_A)

    def smooth_gradient(self, b):
        return self.smooting_solver(b)

gradient_solver = GradientSmoother(A_smooth, dofs=solver.dofs)


### Placeholder for the functions u and p in the rhs 
u_placeholder = Function(solver.V)
p_placeholder = Function(solver.V)

L = - ufl.inner(ufl.grad(u_placeholder), ufl.grad(p_placeholder)) * solver.phi * ufl.dx
b = create_vector(form(L))

num_iter = 1000

last_sigma_j = None 
last_gradient = None 
loss_vals = []

s = 1e-10
step_min = 1e-6

relative_change_list = [] 
stopping_criterion = 1e-3
obs_weight = 40

sigma_background = Function(solver.V)
sigma_background.interpolate(lambda x: np.ones_like(x[0])*sigma0)

sigma_iter = Function(solver.V) 

for step in range(num_iter+1):
    full_time1 = time.time() 
    print("STEP: ", step)

    # compute sigma_j
    sigma_j = Function(solver.V) 
    sigma_j.x.array[:] = sigma_background.x.array[:]  + sigma_iter.x.array[:]

    # simulate forward 
    u_list, U = solver.forward_solve(sigma_j)
    U = np.asarray(U) 

    # compute adjoint
    deltaU = U - Umeas
    p_list = solver.solve_adjoint(deltaU)
    
    loss_l2 = np.sum((U - Umeas)**2)
    
    scalar_h1 = (ufl.inner(ufl.grad(sigma_iter), ufl.grad(solver.phi)) + sigma_iter * solver.phi) * ufl.dx
    b_scalar_h1 = assemble_vector(form(scalar_h1))
    loss_l1 = np.sum(np.abs(b_scalar_h1.array[:]))

    loss = obs_weight*loss_l2 + alpha * loss_l1

    #print("L2 + alpha L1: ", loss)
    print(f"L2 Loss {obs_weight*loss_l2} || L1 Loss {alpha * loss_l1}")
    loss_vals.append(loss)

    # compute smoothed gradients 
    Dsigma_sum = Function(solver.V)
    Dsigma = Function(solver.V)
    for i in range(len(u_list)):
        # Reset right hand side
        with b.localForm() as loc_b:
            loc_b.set(0)
        
        u_placeholder.x.array[:] = u_list[i]
        p_placeholder.x.array[:] = p_list[i]

        # Assemble new rhs and apply Dirichlet boundary condition to the vector
        assemble_vector(b, form(L))
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        # Solve system and add gradient
        #print(b.getArray())
        #gradient_solver.solve(b, Dsigma.vector)
        Dsigma = gradient_solver.smooth_gradient(np.array(b.getArray()))

        Dsigma_sum.x.array[:] += Dsigma 

    Dsigma_sum.x.array[:] = Dsigma_sum.x.array[:] 
    
    if step > 0:
        diff_sigma = sigma_iter - last_sigma_j
        diff_gradient = Dsigma_sum - last_gradient
        t1 = (ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_gradient)) + diff_sigma * diff_gradient) * ufl.dx
        t2 = (ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_sigma)) + diff_sigma * diff_sigma) * ufl.dx
        t1_scalar = np.abs(assemble_scalar(form(t1)))
        t2_scalar = np.abs(assemble_scalar(form(t2)))

        step_size = 4*t2_scalar/t1_scalar
        
        iter_ = 0
        while True: 
            # compute new delta sigma 
            sigma_guess = Function(solver.V) 
            sigma_guess.x.array[:] = sigma_iter.x.array[:] - step_size*Dsigma_sum.x.array[:]

            # compute soft threshold 
            s_dof = np.array(sigma_guess.x.array[:])
            threshold = step_size*alpha

            smaller_idx = (np.abs(s_dof) < threshold)
            bigger_idx = np.invert(smaller_idx)

            s_dof[smaller_idx] = 0.
            s_dof[bigger_idx] -= threshold * np.sign(s_dof[bigger_idx]) 

            s_dof = np.clip(s_dof, l1 - sigma0, l2 - sigma0)
            sigma_guess.x.array[:] = s_dof
            # end of soft threshold

            # sigma_guess and sigma_iter are piecewise constant 
            diff_sigma = sigma_guess - sigma_iter
            diff_sigma_h1 = (ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_sigma)) + diff_sigma * diff_sigma) * ufl.dx
            second_term = assemble_scalar(form(diff_sigma_h1))

            #second_term = np.linalg.norm(cell_area*(sigma_guess.x.array[:] - sigma_iter.x.array[:]))

            sigma_test = Function(solver.V) 
            sigma_test.x.array[:] = sigma_background.x.array[:] + sigma_guess.x.array[:]

            _, Uguess = solver.forward_solve(sigma_test)
            Uguess = np.asarray(Uguess) 

            scalar_h1 = (ufl.inner(ufl.grad(sigma_guess), ufl.grad(solver.phi)) + sigma_guess * solver.phi) * ufl.dx
            b_scalar_h1 = assemble_vector(form(scalar_h1))
            new_loss_l1 = np.sum(np.abs(b_scalar_h1.array[:]))
            #new_loss_l1 = np.sum(cell_area*np.abs(sigma_guess.x.array[:]))
            new_loss_l2 = np.sum((Uguess - Umeas)**2)
            new_loss = obs_weight*new_loss_l2 + alpha * new_loss_l1
            
            # loss_vals is a list of the loss [J_alpha(sigma_k)] for the last M sigma_k's 
            if all([new_loss + step_size*s*second_term > l for l in loss_vals]):
                step_size = step_size/2.

                if step_size < step_min:
                    step_size = step_min
                    break

                continue
            else:
                break
            
        print("Final step size: ", step_size)
    else:
        step_size = 0.0001

    # always keep last N loss values in memory. Needed to the step size computation
    if len(loss_vals) > 5:
        loss_vals.pop(0)

    last_sigma_j = Function(solver.V) 
    last_sigma_j.interpolate(sigma_iter)
    
    sigma_iter.x.array[:] = sigma_iter.x.array[:] - step_size*Dsigma_sum.x.array[:]

    # perform soft threshold 
    s_dof = np.array(sigma_iter.x.array[:])
    threshold = step_size*alpha

    smaller_idx = (np.abs(s_dof) < threshold)
    bigger_idx = np.invert(smaller_idx)

    s_dof[smaller_idx] = 0.
    s_dof[bigger_idx] -= threshold * np.sign(s_dof[bigger_idx]) 

    s_dof = np.clip(s_dof, l1 - sigma0, l2 - sigma0)#
    sigma_iter.x.array[:] = s_dof


    last_gradient = Function(solver.V) 
    last_gradient.interpolate(Dsigma_sum)   

    # stopping criterion 
    diff_sigma = sigma_iter - last_sigma_j
    diff_sigma_h1 = (ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_sigma)) + diff_sigma * diff_sigma) * ufl.dx
    diff_sigma_scalar = assemble_scalar(form(diff_sigma_h1))

    norm_sigma_h1 = (ufl.inner(ufl.grad(sigma_iter), ufl.grad(sigma_iter)) + sigma_iter * sigma_iter) * ufl.dx
    norm_sigma_scalar = assemble_scalar(form(norm_sigma_h1))

    s = diff_sigma_scalar/(norm_sigma_scalar + 1e-4)
    print("||sigma_last - sigma||_H1 / ||sigma||_H1 = ", s)
    relative_change_list.append(s)

    if all([change < stopping_criterion for change in relative_change_list]):
        print("STOPPING CRITERION REACHED AT ITERATION ", step)
        break 

    if len(relative_change_list) > 10:
        relative_change_list.pop(0)

    sigma_j = Function(solver.V)
    sigma_j.x.array[:] = sigma_background.x.array[:]  + sigma_iter.x.array[:]


    if step % 10 == 0 and step > 0:
        
        pred = np.array(sigma_j.x.array[:]).flatten()

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        im = ax.tripcolor(tri, pred, cmap='jet', shading='gouraud',vmin=0.01, vmax=2.0)
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Prediction")
        fig.colorbar(im, ax=ax)

        plt.show()



pred = np.array(sigma_j.x.array[:]).flatten()

fig, ax = plt.subplots(1,1, figsize=(6,6))
im = ax.tripcolor(tri, pred, cmap='jet', shading='gouraud',vmin=0.01, vmax=2.0)
ax.axis('image')
ax.set_aspect('equal', adjustable='box')
ax.set_title("Prediction")
fig.colorbar(im, ax=ax)

plt.show()
