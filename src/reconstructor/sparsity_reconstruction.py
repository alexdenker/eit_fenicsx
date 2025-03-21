"""
min_{delta_s} 1/2 || F(s_background + delta_s) - U ||_2^2 + alpha || delta_s||_1

Sparsity Reconstruction, see
    M. Gehre et al. (2012) "Sparsity reconstruction in electrical impedance tomography: An experimental evaluation",
    Journal of Computational and Applied Mathematics https://www.sciencedirect.com/science/article/pii/S0377042711005140

The class L1Sparsity follows algorithm 2 in the paper:

Let sigma0 be the (known) background conductivity [Line 126]
Set  delta sigma^0 = 0 [Line 135]
for j = 1, ..., J do
    Compute sigma^j = sigma0 + delta sigma^j  [Line 162]
    Compute the gradient D'(sigma^j) by the adjoint method [Line 179]
    Compute the smoothed gradient Ds'(sigma^j) [Line 179]
    Determine step size tau_j [Line 183]
    Update inhomogeneity by delta sigma^j+1 = delta sigma^j - tau_j Ds'(sigma^j) [Line 207]
    Threshold delta sigma^J+1 by S_{tau_j alpha}(delta sigma^j+1) [Line 212]
    Check stopping criterion [Line 231]
end for
Output sigma0 + delta sigma

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

from dolfinx import default_scalar_type
from dolfinx.mesh import exterior_facet_indices
from dolfinx.fem import (
    Constant,
    Function,
    form,
    dirichletbc,
    locate_dofs_topological,
    assemble_scalar,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_vector,
)

from petsc4py import PETSc
import ufl
from tqdm import tqdm

from src.forward_model import EIT
from src.reconstructor import Reconstructor


class GradientSmoother:
    def __init__(self, A_smooth, dofs, backend="Scipy"):
        if backend == "Scipy":
            ai, aj, av = A_smooth.getValuesCSR()
            scipy_A = csr_matrix((av, aj, ai))
            scipy_A.resize(dofs, dofs)

            self.smooting_solver = factorized(scipy_A)

    def smooth_gradient(self, b):
        return self.smooting_solver(b)


class L1Sparsity(Reconstructor):
    def __init__(
        self,
        eit_solver: EIT,
        backCond: float = 1.0,
        kappa: float = 0.01,
        clip=[0.001, 3.0],
        max_iter=200,
        stopping_criterion=5e-4,
        step_min=1e-6,
        s=1e-10,
        initial_step_size=0.25,
        alpha=7e-4,
        **kwargs,
    ):
        super().__init__(eit_solver)

        self.backCond = backCond
        self.L = self.eit_solver.L
        self.kappa = kappa
        self.max_iter = max_iter
        self.stopping_criterion = stopping_criterion
        self.step_min = step_min
        self.s = s
        self.initial_step_size = initial_step_size
        self.alpha = alpha
        # smallest / biggest allowed conductivity
        self.l1 = clip[0]
        self.l2 = clip[1]

        ### set up gradient smoothing ###
        kappa = Constant(self.eit_solver.omega, default_scalar_type(self.kappa))
        fdim = self.eit_solver.omega.topology.dim - 1
        boundary_facets = exterior_facet_indices(self.eit_solver.omega.topology)
        self.bc = dirichletbc(
            PETSc.ScalarType(0),
            locate_dofs_topological(self.eit_solver.V, fdim, boundary_facets),
            self.eit_solver.V,
        )

        # create the LHS matrix
        a = (
            ufl.inner(
                kappa * ufl.grad(self.eit_solver.u), ufl.grad(self.eit_solver.phi)
            )
            + self.eit_solver.u * self.eit_solver.phi
        ) * ufl.dx
        self.bilinear_form = form(a)
        A_smooth = assemble_matrix(self.bilinear_form, bcs=[self.bc])
        A_smooth.assemble()

        self.gradient_solver = GradientSmoother(A_smooth, dofs=self.eit_solver.dofs)

        self.sigma_background = Function(self.eit_solver.V)
        self.sigma_background.interpolate(lambda x: self.backCond * np.ones_like(x[0]))

    def forward(self, Umeas, **kwargs):
        """
        Umeas: numpy array [num_pattern, num_electrodes]

        """
        verbose = kwargs.get("verbose", False)

        sigma_iter = Function(self.eit_solver.V)

        # We re-scale alpha by the number of current pattern to make the same alpha perform well for different pattern sizes
        alpha = Umeas.shape[0] * self.alpha

        ### Placeholder for the functions u and p in the rhs
        u_placeholder = Function(self.eit_solver.V)
        p_placeholder = Function(self.eit_solver.V)

        L = (
            -ufl.inner(ufl.grad(u_placeholder), ufl.grad(p_placeholder))
            * self.eit_solver.phi
            * ufl.dx
        )
        b = create_vector(form(L))

        last_sigma = None
        last_gradient = None
        loss_vals = []

        full_loss_list = []

        relative_change_list = []
        disable = not verbose
        with tqdm(total=self.max_iter, disable=disable) as pbar:
            for step in range(self.max_iter):
                sigma_j = Function(self.eit_solver.V)
                sigma_j.x.array[:] = (
                    self.sigma_background.x.array[:] + sigma_iter.x.array[:]
                )

                u_list, U = self.eit_solver.forward_solve(sigma_j)
                U = np.asarray(U)

                # compute adjoint
                deltaU = U - Umeas
                p_list = self.eit_solver.solve_adjoint(deltaU)

                loss_l2 = np.sum((U - Umeas) ** 2)
                loss_l1 = self.h1_norm(sigma_iter)
                loss = loss_l2 + alpha * loss_l1
                loss_vals.append(loss)
                full_loss_list.append(loss)

                Dsigma_sum = self.compute_gradient(
                    u_list, p_list, b, L, u_placeholder, p_placeholder
                )

                if step > 0:
                    step_size = self.compute_step_size(
                        sigma_iter,
                        Dsigma_sum,
                        last_sigma,
                        last_gradient,
                        alpha,
                        Umeas,
                        loss_vals,
                    )
                else:
                    step_size = self.initial_step_size

                # always keep last N loss values in memory. Needed to the step size computation
                if len(loss_vals) > 5:
                    loss_vals.pop(0)

                # We need to save the last sigma and gradient to compute the step size
                last_sigma = Function(self.eit_solver.V)
                last_sigma.interpolate(sigma_iter)

                last_gradient = Function(self.eit_solver.V)
                last_gradient.interpolate(Dsigma_sum)

                sigma_iter.x.array[:] = (
                    sigma_iter.x.array[:] - step_size * Dsigma_sum.x.array[:]
                )

                threshold = step_size * alpha
                sigma_iter = self.soft_threshold(sigma_iter, threshold)

                # We need the relative change to compute the stopping criterion
                diff_sigma = sigma_iter - last_sigma
                diff_sigma_h1 = (
                    ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_sigma))
                    + diff_sigma * diff_sigma
                ) * ufl.dx
                diff_sigma_scalar = assemble_scalar(form(diff_sigma_h1))

                norm_sigma_h1 = (
                    ufl.inner(ufl.grad(sigma_iter), ufl.grad(sigma_iter))
                    + sigma_iter * sigma_iter
                ) * ufl.dx
                norm_sigma_scalar = assemble_scalar(form(norm_sigma_h1))

                s = diff_sigma_scalar / (norm_sigma_scalar + 1e-4)
                relative_change_list.append(s)

                if (
                    all(
                        [
                            change < self.stopping_criterion
                            for change in relative_change_list
                        ]
                    )
                    or s == 0.0
                ):
                    break

                if len(relative_change_list) > 5:
                    relative_change_list.pop(0)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(full_loss_list[-1], 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        print(f"Stopping criterion reached at iteration {step}")
        return sigma_j

    def compute_gradient(self, u_list, p_list, b, L, u_placeholder, p_placeholder):
        # compute smoothed gradients
        Dsigma_sum = Function(self.eit_solver.V)
        Dsigma = Function(self.eit_solver.V)
        for i in range(len(u_list)):
            with b.localForm() as loc_b:
                loc_b.set(0)

            u_placeholder.x.array[:] = u_list[i]
            p_placeholder.x.array[:] = p_list[i]

            # Assemble new rhs and apply Dirichlet boundary condition to the vector
            assemble_vector(b, form(L))
            apply_lifting(b, [self.bilinear_form], [[self.bc]])
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(b, [self.bc])

            Dsigma = self.gradient_solver.smooth_gradient(np.array(b.getArray()))

            Dsigma_sum.x.array[:] += Dsigma

        Dsigma_sum.x.array[:] = Dsigma_sum.x.array[:] / len(u_list)
        return Dsigma_sum

    def h1_norm(self, sigma):
        scalar_h1 = (
            ufl.inner(ufl.grad(sigma), ufl.grad(self.eit_solver.phi))
            + sigma * self.eit_solver.phi
        ) * ufl.dx
        b_scalar_h1 = assemble_vector(form(scalar_h1))
        return np.sum(np.abs(b_scalar_h1.array[:]))

    def compute_step_size(
        self, sigma_iter, Dsigma_sum, last_sigma, last_gradient, alpha, Umeas, loss_vals
    ):
        diff_sigma = sigma_iter - last_sigma
        diff_gradient = Dsigma_sum - last_gradient
        t1 = (
            ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_gradient))
            + diff_sigma * diff_gradient
        ) * ufl.dx
        t2 = (
            ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_sigma))
            + diff_sigma * diff_sigma
        ) * ufl.dx

        t1_scalar = np.abs(assemble_scalar(form(t1)))
        t2_scalar = np.abs(assemble_scalar(form(t2)))

        step_size = 4 * t2_scalar / t1_scalar

        sigma_test = Function(self.eit_solver.V)
        while True:
            # compute new delta sigma
            sigma_guess = Function(self.eit_solver.V)
            sigma_guess.x.array[:] = (
                sigma_iter.x.array[:] - step_size * Dsigma_sum.x.array[:]
            )

            threshold = step_size * alpha
            sigma_guess = self.soft_threshold(sigma_guess, threshold)

            diff_sigma = sigma_guess - sigma_iter
            diff_sigma_h1 = (
                ufl.inner(ufl.grad(diff_sigma), ufl.grad(diff_sigma))
                + diff_sigma * diff_sigma
            ) * ufl.dx
            second_term = assemble_scalar(form(diff_sigma_h1))

            sigma_test.x.array[:] = (
                self.sigma_background.x.array[:] + sigma_guess.x.array[:]
            )

            _, Uguess = self.eit_solver.forward_solve(sigma_test)
            Uguess = np.asarray(Uguess)

            new_loss_l1 = self.h1_norm(sigma_guess)
            new_loss_l2 = np.sum((Uguess - Umeas) ** 2)
            new_loss = new_loss_l2 + alpha * new_loss_l1

            # loss_vals is a list of the loss [J_alpha(sigma_k)] for the last M sigma_k's
            if all(
                [new_loss + step_size * self.s * second_term > l for l in loss_vals]
            ):
                step_size = step_size / 2.0

                if step_size < self.step_min:
                    step_size = self.step_min
                    break

                continue
            else:
                break

        return step_size

    def soft_threshold(self, sigma, threshold):
        s_dof = np.array(sigma.x.array[:])

        smaller_idx = np.abs(s_dof) < threshold
        bigger_idx = np.invert(smaller_idx)

        s_dof[smaller_idx] = 0.0
        s_dof[bigger_idx] -= threshold * np.sign(s_dof[bigger_idx])

        s_dof = np.clip(s_dof, self.l1 - self.backCond, self.l2 - self.backCond)
        sigma.x.array[:] = s_dof

        return sigma

    def visualise(self):
        xy = self.eit_solver.omega.geometry.x
        cells = self.eit_solver.omega.geometry.dofmap.reshape(
            (-1, self.eit_solver.omega.topology.dim + 1)
        )
        tri = Triangulation(xy[:, 0], xy[:, 1], cells)
