"""
Implementation of a simple Gauss-Newton solver for the CEM. In each iteration we have to solve a linear system of equation. This is solved on the GPU using pytorch. 

"""

import numpy as np 
from tqdm import tqdm 

import torch 
from dolfinx.fem import Function

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

        disable = not verbose
        for i in tqdm(range(num_steps), disable=disable):
            sigma_k = Function(self.eit_solver.V_sigma)
            sigma_k.x.array[:] = sigma

    
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
                    A = A + lamb*torch.diag(torch.diag(A))
                else:
                    A = A + lamb*R

            delta_sigma = torch.linalg.solve(A,b).cpu().numpy()

            # TODO: Implement a good step size search
            step_sizes = np.linspace(0.01, 1, 6)
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

        return sigma