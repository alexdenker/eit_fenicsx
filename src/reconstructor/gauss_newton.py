"""
Implementation of a simple Gauss-Newton solver for the CEM. In each iteration we have to solve a linear system of equation. This is solved on the GPU using pytorch.

"""

import numpy as np
from tqdm import tqdm

import torch
from dolfinx.fem import Function
from scipy.sparse import csr_array

from src.forward_model import EIT
from src.reconstructor import Reconstructor

class GaussNewtonSolver(Reconstructor):
    def __init__(
        self,
        eit_solver: EIT,
        device: str = "cpu",
        num_steps: int = 40,
        R=None,
        lamb: float = 1.0,
        GammaInv: torch.Tensor = None,
        Uel_background: np.array = None,
        clip=[0.001, 3.0],
        backCond: float = 1.0,
    ):
        super().__init__(eit_solver)

        self.device = device

        self.num_steps = num_steps
        self.R = R
        self.lamb = lamb
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.clip = clip
        self.backCond = backCond

    def forward(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        verbose = kwargs.get("verbose", False)
        sigma_init = kwargs.get("sigma_init", None)

        if sigma_init is None:
            sigma_init = Function(self.eit_solver.V_sigma)
            sigma_init.x.array[:] = self.backCond

        if isinstance(self.R, str):
            if self.R == "Tikhonov":
                R = torch.eye(len(sigma_init.x.array[:]), device=self.device)
            elif self.R == "LM":
                pass
            else:
                raise ValueError(f"Unknown string for R: {R}. Choices [Tikhonov, LM]")
        elif isinstance(self.R, torch.Tensor):
            R = self.R.to(self.device)

        if self.GammaInv is not None:
            self.GammaInv = self.GammaInv.to(self.device)

        sigma = sigma_init.x.array[:]

        sigma_old = Function(self.eit_solver.V_sigma)

        disable = not verbose
        with tqdm(total=self.num_steps, disable=disable) as pbar:
            for i in range(self.num_steps):
                sigma_k = Function(self.eit_solver.V_sigma)
                sigma_k.x.array[:] = sigma

                sigma_old.x.array[:] = sigma

                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()

                J = self.eit_solver.calc_jacobian(sigma_k, u_all)

                if self.Uel_background is not None and i == 0:
                    deltaU = self.Uel_background.flatten() - Umeas
                else:
                    deltaU = Usim - Umeas

                J = torch.from_numpy(J).float().to(self.device)
                deltaU = torch.from_numpy(deltaU).float().to(self.device)

                if self.GammaInv is not None:
                    A = J.T @ torch.diag(self.GammaInv) @ J
                    b = J.T @ torch.diag(self.GammaInv) @ deltaU
                else:
                    A = J.T @ J
                    b = J.T @ deltaU

                if R is not None:
                    if R == "LM":
                        A = (
                            A
                            + self.lamb * torch.diag(torch.diag(A))
                            + self.lamb
                            / 2.0
                            * torch.eye(len(sigma_init.x.array[:]), device=self.device)
                        )
                    else:
                        A = A + self.lamb * R

                delta_sigma = torch.linalg.solve(A, b).cpu().numpy()

                # TODO: Implement a good step size search
                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size * delta_sigma

                    sigma_new = np.clip(sigma_new, self.clip[0], self.clip[1])

                    sigmanew = Function(self.eit_solver.V_sigma)
                    sigmanew.x.array[:] = sigma_new

                    _, Utest = self.eit_solver.forward_solve(sigmanew)
                    Utest = np.asarray(Utest).flatten()
                    losses.append(np.sum((Utest - Umeas) ** 2))

                step_size = step_sizes[np.argmin(losses)]

                sigma = sigma + step_size * delta_sigma

                sigma = np.clip(sigma, self.clip[0], self.clip[1])

                s = np.linalg.norm(sigma - sigma_old.x.array[:]) / np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco

    



class LinearisedReconstruction(Reconstructor):
    def __init__(
        self,
        eit_solver: EIT,
        device: str = "cpu",
        R=None,
        lamb: float = 1.0,
        GammaInv: torch.Tensor = None,
        Uel_background: np.array = None,
        clip=[0.001, 3.0],
        backCond: float = 1.0,
    ):
        super().__init__(eit_solver)

        self.device = device

        self.R = R
        self.lamb = lamb
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.clip = clip
        self.backCond = backCond

        self.J = self.calculate_jacobian()
        self.J = torch.from_numpy(self.J).float().to(self.device)
        print("Shape of Jacobian: ", self.J.shape)

    def calculate_jacobian(self):
        sigma_k = Function(self.eit_solver.V_sigma)
        sigma_k.x.array[:] = self.backCond

        u_all, Usim = self.eit_solver.forward_solve(sigma_k)
        Usim = np.asarray(Usim).flatten()

        J = self.eit_solver.calc_jacobian(sigma_k, u_all)

        return J

    def forward(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        lamb = kwargs.get("lamb", None)
        if lamb is None:
            lamb = self.lamb
            #print(f"No regularisation was specified, use {lamb}")

        # if isinstance(self.R, str):
        #    if self.R == "Tikhonov":
        #        R = torch.eye(self.J.shape[1], device=self.device)
        #    elif self.R == "LM":
        #        pass
        #    else:
        #        raise ValueError(f"Unknown string for R: {self.R}. Choices [Tikhonov, LM]")
        # elif isinstance(self.R, torch.Tensor):
        #    R = self.R.to(self.device)

        if self.GammaInv is not None:
            self.GammaInv = self.GammaInv.to(self.device)

        deltaU = self.Uel_background.flatten() - Umeas
        deltaU = torch.from_numpy(deltaU).float().to(self.device)

        if self.GammaInv is not None:
            A = self.J.T @ torch.diag(self.GammaInv) @ self.J
            b = self.J.T @ torch.diag(self.GammaInv) @ deltaU
        else:
            A = self.J.T @ self.J
            b = self.J.T @ deltaU

        # if self.R is not None:
        #    if isinstance(self.R, str):
        #        if self.R == "LM":
        #            A = (
        #                A
        #                + lamb * torch.diag(torch.diag(A))
        #                + lamb
        #                / 2.0
        #                * torch.eye(self.J.shape[1], device=self.device)
        #            )
        #    else:
        #        A = A + lamb * R
        A = A + lamb * self.R.to(self.device)

        delta_sigma = torch.linalg.solve(A, b).cpu().numpy()
        sigma = self.backCond + delta_sigma

        sigma = np.clip(sigma, self.clip[0], self.clip[1])

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco



class GaussNewtonSolverTV(Reconstructor):
    def __init__(
        self,
        eit_solver: EIT,
        device: str = "cpu",
        num_steps: int = 8,
        lamb: float = 0.04,
        beta: float = 1e-6,
        GammaInv: torch.Tensor = None,
        Uel_background: np.array = None,
        clip=[0.001, 3.0],
        **kwargs,
    ):
        super().__init__(eit_solver)

        self.device = device

        self.Ltv = self.construct_tv_matrix()

        coo = self.Ltv.tocoo()  # Convert to COO format
        indices = torch.tensor([coo.row, coo.col], dtype=torch.int64)  # Indices of non-zero elements
        values = torch.tensor(coo.data, dtype=torch.float32)  # Non-zero values

        # Create PyTorch sparse tensor
        self.Ltv_torch = torch.sparse_coo_tensor(indices, values, coo.shape, device=self.device)



        self.num_steps = num_steps
        self.lamb = lamb
        self.beta = beta
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.clip = clip

    def construct_tv_matrix(self):
        self.eit_solver.omega.topology.create_connectivity(
            1, 2
        )  # Facet-to-cell connectivity
        self.eit_solver.omega.topology.create_connectivity(
            2, 1
        )  # Cell-to-facet connectivity

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

        return csr_array((data, (rows, cols)), shape=(row_idx, num_cells))
        # CUDA does currently not really support CSR tensors
        # return torch.sparse_csr_tensor(torch.tensor(rows), torch.tensor(cols), torch.tensor(data), dtype=torch.float64,size=(row_idx, num_cells))

    def forward(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        verbose = kwargs.get("verbose", False)
        sigma_init = kwargs.get("sigma_init", None)

        if sigma_init is None:
            sigma_init = Function(self.eit_solver.V_sigma)
            sigma_init.x.array[:] = self.backCond

        if self.GammaInv is not None:
            GammaInv = self.GammaInv.to(self.device)
        else:
            GammaInv = torch.ones(Umeas.shape, device=self.device)
            
        sigma = sigma_init.x.array[:]

        sigma_old = Function(self.eit_solver.V_sigma)

        disable = not verbose
        with tqdm(total=self.num_steps, disable=disable) as pbar:
            for i in range(self.num_steps):
                sigma_k = Function(self.eit_solver.V_sigma)
                sigma_k.x.array[:] = sigma

                sigma_old.x.array[:] = sigma

                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()

                J = self.eit_solver.calc_jacobian(sigma_k, u_all)

                if self.Uel_background is not None and i == 0:
                    deltaU = self.Uel_background.flatten() - Umeas
                else:
                    deltaU = Usim - Umeas

                J = torch.from_numpy(J).float().to(self.device)
                deltaU = torch.from_numpy(deltaU).float().to(self.device)

                if GammaInv is not None:
                    A = J.T @ torch.diag(GammaInv) @ J
                    b = J.T @ torch.diag(GammaInv) @ deltaU
                else:
                    A = J.T @ J
                    b = J.T @ deltaU

                """
                L_sigma = np.abs(self.Ltv @ np.array(sigma_k.x.array[:])) ** 2
                eta = np.sqrt(L_sigma + self.beta)
                E = np.diag(1 / eta)

                A = A + torch.from_numpy(
                    self.lamb * self.Ltv.T @ E @ self.Ltv
                ).float().to(self.device)
                b = b - torch.from_numpy(
                    self.lamb * self.Ltv.T @ E @ self.Ltv @ sigma_k.x.array[:]
                ).float().to(self.device)
                """

                sigma_k_torch = torch.tensor(sigma_k.x.array[:], device=self.device, dtype=torch.float32)

                L_sigma = torch.abs(self.Ltv_torch @ sigma_k_torch) ** 2
                eta = torch.sqrt(L_sigma + self.beta)
                E = torch.diag(1 / eta)

                A = A + self.lamb * self.Ltv_torch.T @ E @ self.Ltv_torch
                b = b - self.lamb * self.Ltv_torch.T @ E @ self.Ltv_torch @ sigma_k_torch
                        
                delta_sigma = torch.linalg.solve(A, b).cpu().numpy()

                # TODO: Implement a good step size search
                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size * delta_sigma

                    sigma_new = np.clip(sigma_new, self.clip[0], self.clip[1])

                    sigmanew = Function(self.eit_solver.V_sigma)
                    sigmanew.x.array[:] = sigma_new

                    _, Utest = self.eit_solver.forward_solve(sigmanew)
                    Utest = np.asarray(Utest).flatten()

                    tv_value = self.lamb * np.sqrt(((self.Ltv @ sigma_new) ** 2) + self.beta).sum()
                    #print(GammaInv.shape, Umeas.shape)

                    meas_value = 0.5 * np.sum((torch.diag(GammaInv).cpu().numpy() @ (Utest - Umeas)) ** 2)
                    #print(meas_value, tv_value)
                    #losses.append(np.sum((Utest - Umeas) ** 2))
                    losses.append(meas_value + tv_value)

                step_size = step_sizes[np.argmin(losses)]

                sigma = sigma + step_size * delta_sigma

                sigma = np.clip(sigma, self.clip[0], self.clip[1])

                s = np.linalg.norm(sigma - sigma_old.x.array[:]) / np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco


    def single_step(self, sigma, Umeas):
        sigma_k = Function(self.eit_solver.V_sigma)
        sigma_k.x.array[:] = sigma

        u_all, Usim = self.eit_solver.forward_solve(sigma_k)
        Usim = torch.tensor(np.asarray(Usim).flatten(), device=self.device, dtype=torch.float32)
        Umeas = torch.tensor(Umeas, device=self.device, dtype=torch.float32)

        J = torch.tensor(self.eit_solver.calc_jacobian(sigma_k, u_all),
                         device=self.device,
                         dtype=torch.float32)

        deltaU = Usim - Umeas

        if self.GammaInv is not None:
            A = J.T @ torch.diag(self.GammaInv) @ J
            b = J.T @ torch.diag(self.GammaInv) @ deltaU
        else:
            A = J.T @ J
            b = J.T @ deltaU

        sigma_k_torch = torch.tensor(sigma_k.x.array[:], device=self.device, dtype=torch.float32)

        L_sigma = torch.abs(self.Ltv_torch @ sigma_k_torch) ** 2
        eta = torch.sqrt(L_sigma + self.beta)
        E = torch.diag(1 / eta)

        A = A + self.lamb * self.Ltv_torch.T @ E @ self.Ltv_torch
        b = b - self.lamb * self.Ltv_torch.T @ E @ self.Ltv_torch @ sigma_k_torch

        delta_sigma = torch.linalg.solve(A, b)

        return delta_sigma