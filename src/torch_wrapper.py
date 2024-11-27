

import torch
import torch.nn as nn 
import numpy as np 

from dolfinx import default_scalar_type
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem import (Constant, Function, FunctionSpace, form, dirichletbc, 
                         locate_dofs_topological)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector
from petsc4py import PETSc
import ufl

class CEMModule(nn.Module):
    def __init__(self, eit_solver, kappa=5e-3, gradient_smooting=False):
        super(CEMModule, self).__init__()
        self.cem_function = CEMTorch()

        self.gradient_smooting = gradient_smooting

        self.eit_solver = eit_solver
        
        #self.function_space = eit_solver.V_sigma
        self.function_space = eit_solver.V

        self.kappa = Constant(self.eit_solver.omega, default_scalar_type(kappa)) 

        ### Create boundary condition
        fdim = self.eit_solver.omega.topology.dim - 1
        boundary_facets = exterior_facet_indices(self.eit_solver.omega.topology)
        self.bc = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(self.eit_solver.V, fdim, boundary_facets), self.eit_solver.V)

        if self.gradient_smooting:
            a = (ufl.inner(self.kappa * ufl.grad(self.eit_solver.u), ufl.grad(self.eit_solver.phi))  + self.eit_solver.u * self.eit_solver.phi) * ufl.dx
        else:
            a = self.eit_solver.u * self.eit_solver.phi * ufl.dx

        self.bilinear_form = form(a)
        if self.gradient_smooting:
            A_smooth = assemble_matrix(self.bilinear_form, bcs=[self.bc])
        else:
            A_smooth = assemble_matrix(self.bilinear_form)

        A_smooth.assemble()

        ### Build solvers
        gradient_solver = PETSc.KSP().create(self.eit_solver.omega.comm)
        gradient_solver.setOperators(A_smooth)
        gradient_solver.setType(PETSc.KSP.Type.PREONLY)
        gradient_solver.getPC().setType(PETSc.PC.Type.LU)

        # dummy rhs
        L = self.eit_solver.phi * ufl.dx
        self.b = create_vector(form(L))

        self.gradient_solver = gradient_solver

    def forward(self, sigma):

        return self.cem_function.apply(sigma, self.eit_solver, self.function_space, self.gradient_solver, self.b, self.bilinear_form, self.bc,self.gradient_smooting)


class CEMTorch(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, sigma_torch, eit_solver, function_space, gradient_solver, b, bilinear_form, bc, gradient_smooting):
        ctx.set_materialize_grads(False)
        ctx.sigma_torch = sigma_torch
        ctx.eit_solver = eit_solver
        ctx.function_space = function_space
        ctx.gradient_solver = gradient_solver
        ctx.bilinear_form = bilinear_form 
        ctx.b = b 
        ctx.bc = bc 
        ctx.gradient_smooting = gradient_smooting
        # ctx is a context object that can be used to stash information

        # sigma_torch [batch, x]
        batches = sigma_torch.shape[0]
        sigma_np = ctx.sigma_torch.detach().cpu().numpy() # get our image as a numpy array

        val = torch.zeros(batches, eit_solver.Inj.shape[0], eit_solver.L, dtype=torch.double, device=sigma_torch.device)
        ctx.us = [] 
        for b in range(batches):
            sigma_fenics = Function(ctx.function_space)
            sigma_fenics.x.array[:] = sigma_np[b]
            u, tmp = eit_solver.forward_solve(sigma_fenics)
            val[b] = torch.tensor(np.array(tmp), dtype=torch.double, device=sigma_torch.device)
            ctx.us.append(u)
        return val
 
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # ctx is a context object that can be used to stash information
        # sigma_torch 3D [batch, 76, 32]
        batches = ctx.sigma_torch.shape[0]
        grad_torch = torch.zeros(ctx.sigma_torch.shape, dtype=torch.double,device=ctx.sigma_torch.device)

        eit_solver = ctx.eit_solver
        function_space = ctx.function_space

        sigma_np = ctx.sigma_torch.detach().cpu().numpy()
        for batch in range(batches):
            delta_u_fenics = grad_output[batch].detach().cpu().numpy()
           
            #sigma_fenics = Function(function_space)
            #sigma_fenics.x.array[:]= sigma_np[batch]

            p_list = eit_solver.solve_adjoint(delta_u_fenics)#, sigma_fenics)

            Dsigma_sum = Function(eit_solver.V)
            u_placeholder = Function(eit_solver.V)
            p_placeholder = Function(eit_solver.V)
            for i in range(len(p_list)):
                Dsigma = Function(eit_solver.V)

                with ctx.b.localForm() as loc_b:
                    loc_b.set(0)
    
                L = - ufl.inner(ufl.grad(u_placeholder), ufl.grad(p_placeholder)) * eit_solver.phi * ufl.dx

                u_placeholder.x.array[:] = ctx.us[batch][i]
                p_placeholder.x.array[:] = p_list[i]

                # Assemble new rhs and apply Dirichlet boundary condition to the vector
                assemble_vector(ctx.b, form(L))

                if ctx.gradient_smooting:
                    apply_lifting(ctx.b, [ctx.bilinear_form], [[ctx.bc]])
                    ctx.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
                    set_bc(ctx.b, [ctx.bc])

                ctx.gradient_solver.solve(ctx.b, Dsigma.vector)

                Dsigma_sum.x.array[:] += Dsigma.x.array[:]

            sigma_update = Function(function_space)
            sigma_update.interpolate(Dsigma_sum)

            sigma_update_np = np.array(sigma_update.x.array[:])/len(p_list)
            grad_torch[batch] = torch.tensor(sigma_update_np, dtype=torch.double,device=ctx.sigma_torch.device)

        return grad_torch, None, None, None, None, None, None, None 


