

import numpy as np 
import time 

from mpi4py import MPI
import ufl 
from dolfinx.io import gmshio
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar, form, Expression, assemble)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

class EIT():
    def __init__(self, Inj, z, backend="PETSc"):
        """
        Inj: current injection pattern (32, N) matrix with the number of patterns N
        z: contact impedance (32)
        """
        assert backend in ["PETSc", "Scipy"], "backend has to be either PETSc or Scipy"

        self.backend = backend

        self.Inj = Inj 
        self.z = z

        self.omega, _, facet_markers = gmshio.read_from_msh("/home/alexdenker/Bremen/dl_eit/FenicsX/Mesh/potential_mesh.msh", MPI.COMM_WORLD, gdim=2)
        #self.omega, _, facet_markers = gmshio.read_from_msh("src/forward_operator/Mesh/EIT_disk_res5.msh", MPI.COMM_WORLD, gdim=2)
        self.omega.topology.create_connectivity(1, 2)
    
        ## Boundary measure
        self.ds_electrodes = ufl.Measure("ds", domain=self.omega, subdomain_data=facet_markers) 

        ## Length of one single electrode (idx. 1-32):
        self.electrode_len = assemble_scalar(form(1 * self.ds_electrodes(1)))

        ### Create function space and helper functions
        self.V = functionspace(self.omega, ("Lagrange", 1))

        u_sol = Function(self.V)
        self.dofs = len(u_sol.x.array)

        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)

        self.M = self.assemble_lhs()

    def assemble_lhs(self):
        ### Construct constant matrix (independent of sigma):
        b = 0
        for i in range(0, 32):
            b += 1/self.z[i] * ufl.inner(self.u, self.phi) * self.ds_electrodes(i+1)

        B = assemble_matrix(form(b))
        B.assemble()

        bi, bj, bv = B.getValuesCSR()
        M = csr_matrix((bv, bj, bi)) # M will be matrix used to solve problem
        M.resize(self.dofs+33, self.dofs+33) # Extra row for zero average condition 
        M_lil = lil_matrix(M) # faster to modify (then change back)

        B.destroy() # dont need B anymore

        for i in range(0, 32):
            # Build C matrix (top right and bottom left block)
            # Has to be done for each row
            c = - 1/self.z[i] * self.phi * self.ds_electrodes(i+1)
            C_i = assemble_vector(form(c)).array
            M_lil[self.dofs+i, :self.dofs] = C_i 
            M_lil[:self.dofs, self.dofs+i] = C_i 
            # bottom right block matrix (diagonal and average condition)
            M_lil[self.dofs+i, self.dofs+i] = 1/self.z[i] * self.electrode_len
            M_lil[self.dofs+32, self.dofs+i] = 1 
            M_lil[self.dofs+i, self.dofs+32] = 1 

        M = csr_matrix(M_lil)
        print("Building background matrix M is done")

        return M

    def create_full_matrix(self, sigma):

        a = ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        A = assemble_matrix(form(a))
        A.assemble()

        ai, aj, av = A.getValuesCSR()
        scipy_A = csr_matrix((av, aj, ai))
        scipy_A.resize(self.dofs+33, self.dofs+33)

        M_complete = scipy_A + self.M
        return M_complete 

    def forward_solve(self, sigma, Inj=None):
        """
        sigma: Fenics functions
        Inj: optional, solve for a different pattern current 
            injection pattern (32, N) matrix with the number of patterns N
        """

        if Inj is None:
            Inj = self.Inj

        ##  build part of LHS dependent on sigma ##

        self.M_complete = self.create_full_matrix(sigma) 
        
        if self.backend == "PETSc":

            petsc_mat = PETSc.Mat().createAIJ(size= self.M_complete.shape, 
                                    csr=( self.M_complete.indptr,  self.M_complete.indices,  self.M_complete.data))
            ## Even if last diagonal entry is already zero (not set in the matrix),
            ## we have to set it per hand to zero:
            petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            petsc_mat.setValues([self.dofs+32], [self.dofs+32], [[0.0]], PETSc.InsertMode.INSERT_VALUES)
            petsc_mat.assemblyBegin()
            petsc_mat.assemblyEnd()
            petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

            solver = PETSc.KSP().create(self.omega.comm)
            solver.setOperators(petsc_mat)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)

            # create empty PETSc vectors for RHS and solution u
            petsc_vec = PETSc.Vec()
            petsc_vec.create(PETSc.COMM_WORLD)
            petsc_vec.setSizes(self.dofs + 33)
            petsc_vec.setUp()

            u_sol_petsc = PETSc.Vec()
            u_sol_petsc.create(PETSc.COMM_WORLD)
            u_sol_petsc.setSizes(self.dofs + 33)
            u_sol_petsc.setUp()

            u_all = [] 
            U_all = [] 
            for inj_idx in range(Inj.shape[-1]):
                with petsc_vec.localForm() as loc_b:
                    loc_b.set(0)

                for i in range(Inj.shape[0]):
                    petsc_vec.setValues([self.dofs + i], [self.Inj[i, inj_idx]])
                    petsc_vec.assemblyBegin()
                    petsc_vec.assemblyEnd()

                solver.solve(petsc_vec, u_sol_petsc)

                sol = np.array(u_sol_petsc.copy().getArray())
                u_all.append(sol[:self.dofs])
                U_all.append(sol[self.dofs:-1])

        elif self.backend == "Scipy":

            solver = factorized(self.M_complete)

            u_all = []
            U_all = [] 

            for inj_idx in range(Inj.shape[-1]):

                rhs = np.zeros(self.dofs+33)
                for i in range(Inj.shape[0]):
                    rhs[self.dofs + i] = Inj[i, inj_idx]
                    
                sol = solver(rhs)
                u_all.append(sol[:self.dofs])
                U_all.append(sol[self.dofs:-1])


        return u_all, U_all 

    def solve_adjoint(self, deltaU, sigma=None):
        """
        deltaU: NxM matrix with:
                    N number of current patterns
                    M number of electrodes
        sigma: Fenicsx function, if None use the one from the forward pass 

        """
       
        if sigma == None:
            M_complete = self.M_complete.copy()

        else: 
            M_complete = self.create_full_matrix(sigma) 

        if self.backend == "PETSc":

            petsc_mat = PETSc.Mat().createAIJ(size= M_complete.shape, 
                                    csr=(M_complete.indptr, M_complete.indices, M_complete.data))
            ## Even if last diagonal entry is already zero (not set in the matrix),
            ## we have to set it per hand to zero:
            petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            petsc_mat.setValues([self.dofs+32], [self.dofs+32], [[0.0]], PETSc.InsertMode.INSERT_VALUES)
            petsc_mat.assemblyBegin()
            petsc_mat.assemblyEnd()
            petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)

            solver = PETSc.KSP().create(self.omega.comm)
            solver.setOperators(petsc_mat)
            solver.setType(PETSc.KSP.Type.PREONLY)
            solver.getPC().setType(PETSc.PC.Type.LU)

            # create empty PETSc vectors for RHS and solution u
            petsc_vec = PETSc.Vec()
            petsc_vec.create(PETSc.COMM_WORLD)
            petsc_vec.setSizes(self.dofs + 33)
            petsc_vec.setUp()

            u_sol_petsc = PETSc.Vec()
            u_sol_petsc.create(PETSc.COMM_WORLD)
            u_sol_petsc.setSizes(self.dofs + 33)
            u_sol_petsc.setUp()

            p_all = [] 
            for inj_idx in range(deltaU.shape[0]): # pattern
                with petsc_vec.localForm() as loc_b:
                    loc_b.set(0)

                for i in range(deltaU.shape[-1]):
                    petsc_vec.setValues([self.dofs + i], [deltaU[inj_idx, i]])
                    petsc_vec.assemblyBegin()
                    petsc_vec.assemblyEnd()

                solver.solve(petsc_vec, u_sol_petsc)

                sol = np.array(u_sol_petsc.copy().getArray())
                p_all.append(sol[:self.dofs])
        
        elif self.backend == "Scipy":

            solver = factorized(M_complete)

            p_all = []
            for inj_idx in range(deltaU.shape[0]):
                
                rhs = np.zeros(self.dofs+33)
                for i in range(deltaU.shape[-1]):
                    rhs[self.dofs + i] = deltaU[inj_idx, i] 

                sol = solver(rhs)
                p_all.append(sol[:self.dofs])

        return p_all
    

    def calc_jacobian(self, sigma):
        """
        Inj: optional, solve for a different pattern current 
            injection pattern (32, N) matrix with the number of patterns N

        """

        u_all, U_all = self.forward_solve(sigma)

        # here we have to solve the forward problem with a different injection pattern 

        #Construction new current pattern for Jacobian calc.
        #Ref: https://fabiomargotti.paginas.ufsc.br/files/2017/12/Margotti_Fabio-3.pdf chap 5.2.1
        I2_all=[]  #We will use to construct a smart way to calc. Jacobian
        for i in range(self.Inj.shape[0]):
            #I2_i=1 at electrode i and zero otherwise
            I2 = np.zeros(self.Inj.shape[0])
            I2[i] = 1
            I2_all.append(I2)
        
        I2_all = np.array(I2_all).T
        print("Size of new injection pattern, should be [32, ...] : ", I2_all.shape)
        print(I2_all)
        bu_all, BU = self.forward_solve(sigma, I2_all)

        
        # Define Q_DG as a Discontinuous Galerkin vector function space
        Q_DG = functionspace(self.omega, ("DG", 0, (2, )))

        DG0 = functionspace(self.omega, ("DG", 0))

        v = ufl.TestFunction(DG0)
        cell_area_form = form(v*ufl.dx)
        cell_area = assemble_vector(cell_area_form)
        #print(cell_area.array)

        #  a(u,v)=inner(u,v)*dx=inner(grad(f),v)*dx=l(v)
        # u,v are Trial and test functions for the space you want to project into

        # Project the gradient of 'u' into Q_DG and reshape the results
        
        """
        list_grad_u = []
        for u in u_all:
            u_fun = Function(self.V)
            u_fun.x.array[:] = u 

            grad_u = Function(Q_DG)
            grad_u_expr = Expression(ufl.as_vector((u_fun.dx(0), u_fun.dx(1))), Q_DG.element.interpolation_points())
            grad_u.interpolate(grad_u_expr)

            grad_u_vec = grad_u.vector.array.reshape(-1, 2)
            list_grad_u.append(grad_u_vec)

        # Project the gradient of 'bu' into Q_DG and reshape the results
        list_grad_bu = []
        for bu in bu_all:
            bu_fun = Function(self.V)
            bu_fun.x.array[:] = bu 

            grad_u = Function(Q_DG)
            grad_u_expr = Expression(ufl.as_vector((bu_fun.dx(0), bu_fun.dx(1))), Q_DG.element.interpolation_points())
            grad_u.interpolate(grad_u_expr)

            grad_u_vec = grad_u.vector.array.reshape(-1, 2)
            list_grad_bu.append(grad_u_vec)

        print(len(list_grad_u), len(list_grad_bu))
        print(len(list_grad_u[0].shape))
        """ 

        u1 = Function(self.V)
        u2 = Function(self.V)

        v = ufl.TestFunction(DG0)
        v_ = Function(DG0)
        print("SHAPES: ", v_.x.array.shape, u1.x.array.shape)
        for h in range(len(u_all)): #For each experiment
            derivative =[]
            u1.x.array[:] = u_all[h] #list_grad_u[h]
            for j in range(32): #for each electrode
                u2.x.array[:] = bu_all[j]  #list_grad_bu[j]

                #row = assemble_vector(form(-1*ufl.inner(u1,u2)*ufl.dx))
                #row = assemble_vector(form(-1*(u1[0] * u2[0] + u1[0] * u2[0])*ufl.dx))
                row = assemble_vector(form(-1*v*ufl.inner(ufl.grad(u1), ufl.grad(u2))*ufl.dx))

                #print(row)
                #row2 = -1*np.sum(list_grad_bu[j]*list_grad_u[h], axis=1)*cell_area.array
                #print(np.sum((row2 - row.array)**2))
                #print(row2[0:10], row.array[0:10])
                derivative.append(row.array) #Get the function value in eache element
                
            Jacobian = np.array(derivative) #Matrix * Volume_cell
            if h==0: 
                Jacobian_all=Jacobian #Append all jacs.
            else: 
                Jacobian_all=np.concatenate((Jacobian_all, Jacobian), axis=0)
            
        return Jacobian_all
