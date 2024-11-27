"""
Implementation of a simple Gauss-Newton solver for the CEM. In each iteration we have to solve a linear system of equation. This is solved on the GPU using pytorch. 

"""

import numpy as np 
from tqdm import tqdm 

import torch 
from dolfinx.fem import Function
from scipy.sparse import csr_array

from src.eit_forward_fenicsx import EIT

class GaussNewtonSolver():
    def __init__(self, eit_solver: EIT, device: str ="cpu"):

        self.eit_solver = eit_solver
        self.device = device


    def reconstruct(self, Umeas: np.array, 
                        sigma_init: Function, 
                        num_steps: int = 40, 
                        R = None, 
                        lamb: float = 1.0,
                        GammaInv: torch.Tensor = None, 
                        verbose: bool = True, 
                        clip=[0.001, 3.0]):
        
        if isinstance(R, str): 
            if R == "Tikhonov":
                R = torch.eye(len(sigma_init.x.array[:]), device=self.device)
            elif R == "LM":
                pass 
            else:
                raise ValueError(f"Unknown string for R: {R}. Choices [Tikhonov, LM]")
        elif isinstance(R, torch.Tensor):
            R = R.to(self.device)

        if GammaInv is not None:
            GammaInv = GammaInv.to(self.device)

        sigma = sigma_init.x.array[:]


        sigma_old = Function(self.eit_solver.V_sigma)

        disable = not verbose
        with tqdm(total=num_steps, disable=disable) as pbar:

            for i in range(num_steps):
                sigma_k = Function(self.eit_solver.V_sigma)
                sigma_k.x.array[:] = sigma

                sigma_old.x.array[:] = sigma
        
                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()

                J = self.eit_solver.calc_jacobian(sigma_k, u_all)

                deltaU = Usim - Umeas

                J = torch.from_numpy(J).float().to(self.device)
                deltaU = torch.from_numpy(deltaU).float().to(self.device)

                if GammaInv is not None:
                    A = J.T @ torch.diag(GammaInv) @ J   
                    b = J.T @ torch.diag(GammaInv) @ deltaU 
                else:
                    A = J.T @ J
                    b = J.T @ deltaU 

                if R is not None:
                    if R == "LM":
                        A = A + lamb*torch.diag(torch.diag(A)) + lamb/2. * torch.eye(len(sigma_init.x.array[:]), device=self.device)
                    else:
                        A = A + lamb*R 

                delta_sigma = torch.linalg.solve(A,b).cpu().numpy()

                # TODO: Implement a good step size search
                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size*delta_sigma

                    sigma_new = np.clip(sigma_new, clip[0], clip[1])

                    sigmanew = Function(self.eit_solver.V_sigma)
                    sigmanew.x.array[:] = sigma_new

                    _, Utest = self.eit_solver.forward_solve(sigmanew)
                    Utest = np.asarray(Utest).flatten()
                    losses.append(np.sum((Utest - Umeas)**2))

                step_size = step_sizes[np.argmin(losses)]

                sigma = sigma + step_size*delta_sigma

                sigma = np.clip(sigma, clip[0], clip[1])

                s = np.linalg.norm(sigma - sigma_old.x.array[:])/np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}")
                pbar.update(1)

        return sigma
    

class GaussNewtonSolverTV():
    def __init__(self, eit_solver: EIT, device: str ="cpu"):

        self.eit_solver = eit_solver
        self.device = device

        self.Ltv = self.construct_tv_matrix()

    def construct_tv_matrix(self):

        self.eit_solver.omega.topology.create_connectivity(1, 2)  # Facet-to-cell connectivity
        self.eit_solver.omega.topology.create_connectivity(2, 1)  # Cell-to-facet connectivity

        # Number of cells in the mesh
        num_cells = self.eit_solver.omega.topology.index_map(2).size_local

        cell_to_edge = self.eit_solver.omega.topology.connectivity(2, 1)
        # cell_to_edge connects a cell to its border, every cell has always three borders because the cell is a triangle 
        # e.g. 0: [4 1 0] "Cell 0 connects to border 4, 1 and 0"

        edge_to_cell = self.eit_solver.omega.topology.connectivity(1, 2)
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
                if len(adjacent_cells) > 1: # only look at the parts where we have two cells 
                    rows.append(row_idx)
                    rows.append(row_idx)
                    cols.append(adjacent_cells[0])
                    cols.append(adjacent_cells[1])
                    data.append(1)
                    data.append(-1)

                    row_idx += 1

        return csr_array((data,(rows, cols)), shape=(row_idx, num_cells))
        # CUDA does currently not really support CSR tensors
        #return torch.sparse_csr_tensor(torch.tensor(rows), torch.tensor(cols), torch.tensor(data), dtype=torch.float64,size=(row_idx, num_cells))

    def reconstruct(self, Umeas: np.array, 
                        sigma_init: Function, 
                        num_steps: int = 40, 
                        lamb: float = 1.0,
                        beta: float = 1.0,
                        GammaInv: torch.Tensor = None, 
                        verbose: bool = True, 
                        clip=[0.001, 3.0]):

        if GammaInv is not None:
            GammaInv = GammaInv.to(self.device)

        sigma = sigma_init.x.array[:]


        sigma_old = Function(self.eit_solver.V_sigma)

        disable = not verbose
        with tqdm(total=num_steps, disable=disable) as pbar:

            for i in range(num_steps):
                sigma_k = Function(self.eit_solver.V_sigma)
                sigma_k.x.array[:] = sigma

                sigma_old.x.array[:] = sigma
        
                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()

                J = self.eit_solver.calc_jacobian(sigma_k, u_all)

                deltaU = Usim - Umeas

                J = torch.from_numpy(J).float().to(self.device)
                deltaU = torch.from_numpy(deltaU).float().to(self.device)

                if GammaInv is not None:
                    A = J.T @ torch.diag(GammaInv) @ J   
                    b = J.T @ torch.diag(GammaInv) @ deltaU 
                else:
                    A = J.T @ J
                    b = J.T @ deltaU 

                L_sigma = np.abs(self.Ltv @ np.array(sigma_k.x.array[:]))**2
                eta = np.sqrt(L_sigma + beta)
                E = np.diag(1/eta)

                A = A + torch.from_numpy(lamb * self.Ltv.T @ E @ self.Ltv).float().to(self.device)
                b = b - torch.from_numpy(lamb * self.Ltv.T @ E @ self.Ltv @ sigma_k.x.array[:]).float().to(self.device)

                delta_sigma = torch.linalg.solve(A,b).cpu().numpy()

                # TODO: Implement a good step size search
                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size*delta_sigma

                    sigma_new = np.clip(sigma_new, clip[0], clip[1])

                    sigmanew = Function(self.eit_solver.V_sigma)
                    sigmanew.x.array[:] = sigma_new

                    _, Utest = self.eit_solver.forward_solve(sigmanew)
                    Utest = np.asarray(Utest).flatten()
                    losses.append(np.sum((Utest - Umeas)**2))

                step_size = step_sizes[np.argmin(losses)]

                sigma = sigma + step_size*delta_sigma

                sigma = np.clip(sigma, clip[0], clip[1])

                s = np.linalg.norm(sigma - sigma_old.x.array[:])/np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}")
                pbar.update(1)

        return sigma