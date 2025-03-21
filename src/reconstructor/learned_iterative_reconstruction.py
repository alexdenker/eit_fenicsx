import os
import yaml
import torch
import torch.nn as nn 
import numpy as np
from dolfinx.fem import Function
from scipy.interpolate import LinearNDInterpolator

from src.reconstructor import Reconstructor
from src.networks import get_unet_model
from src.utils import image_to_mesh


class IterativeBlock(nn.Module):
    def __init__(self, n_in=2, n_out=1, n_layer=3, internal_ch=32,
                 kernel_size=3, batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(IterativeBlock, self).__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        modules = []
        if batch_norm:
            modules.append(nn.BatchNorm2d(n_in))
        for i in range(n_layer-1):
            input_ch = (n_in) if i == 0 else internal_ch
            modules.append(nn.Conv2d(input_ch, internal_ch,
                                     kernel_size=kernel_size, padding=padding))
            if batch_norm:
                modules.append(nn.BatchNorm2d(internal_ch))
            if prelu:
                modules.append(nn.PReLU(internal_ch, init=0.0))
            else:
                modules.append(nn.LeakyReLU(lrelu_coeff, inplace=True))
        modules.append(nn.Conv2d(internal_ch, n_out,
                                 kernel_size=kernel_size, padding=padding))
        self.block = nn.Sequential(*modules)
        self.relu = nn.LeakyReLU(lrelu_coeff, inplace=True)  # remove?

    def forward(self, x):
        upd = self.block(x)
        return upd


class IterativeNet(nn.Module):
    def __init__(self, solver, n_iter, backCond, gn_reconstructor):
        super(IterativeNet, self).__init__()
        self.n_iter = n_iter

        self.solver = solver 
        self.backCond = backCond 
        self.gn_reconstructor = gn_reconstructor 

        self.mesh_pos = np.array(self.solver.V_sigma.tabulate_dof_coordinates()[:, :2])
        self.xy = self.solver.omega.geometry.x
        self.cells = self.solver.omega.geometry.dofmap.reshape(
        (-1, self.solver.omega.topology.dim + 1) )


        self.blocks = nn.ModuleList()
        for it in range(n_iter):
            self.blocks.append(get_unet_model(
                        in_ch=2,
                        out_ch=1,
                        scales=3,
                        skip=4,
                        channels=(32, 32, 64),
                        use_sigmoid=False,
                        use_norm=True))

            #self.blocks.append(IterativeBlock(
            #    n_in=2, n_out=1, n_layer=n_layer,
            #    internal_ch=internal_ch, kernel_size=kernel_size,
            #    batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))

    def freeze_blocks(self, n):
        # Freeze parameters of all blocks up to block n
        for i in range(n + 1):
            for param in self.blocks[i].parameters():
                param.requires_grad = False

    def forward(self, Umeas):
        sigma_reco = Function(self.solver.V_sigma)

        sigma = torch.ones(1, 1, 128, 128, device=Umeas.device) * self.backCond

        for i in range(self.n_iter):
            # This part is not differentiable, so we have to train greedy
            delta_sigma_img = self.compute_delta_sigma(sigma, Umeas) 
            sigma = self.blocks[i](torch.cat([sigma, delta_sigma_img], dim=1))
        
        sigma_mesh = image_to_mesh(sigma[0].cpu().numpy(), mesh_pos=self.mesh_pos)
        sigma_reco.x.array[:] = sigma_mesh
        return sigma_reco

    def compute_delta_sigma(self, sigma, Umeas):
        
        sigma_batched = []
        for i in range(sigma.shape[0]):
            sigma_mesh = image_to_mesh(sigma[0].cpu().numpy(), mesh_pos=self.mesh_pos)
            
            delta1 = 0.1  # noise level
            delta2 = 0.001 
            #var_meas = (delta1 * np.abs(Umeas[i].cpu().numpy().flatten() - self.gn_reconstructor.Uel_background) + delta2 * np.max(np.abs(Umeas[i].cpu().numpy().flatten() - self.gn_reconstructor.Uel_background))) ** 2
            var_meas = (delta1 * np.abs(self.gn_reconstructor.Uel_background) + delta2 * np.max(np.abs(self.gn_reconstructor.Uel_background))) ** 2
            GammaInv = 1.0 / (np.maximum(var_meas.flatten(),1e-5))
            GammaInv = torch.from_numpy(GammaInv).float().to(self.gn_reconstructor.device)

            self.gn_reconstructor.GammaInv = GammaInv
            delta_sigma = self.gn_reconstructor.single_step(sigma_mesh.flatten(), Umeas[i].cpu().numpy())
            delta_sigma_img = self.interpolate_to_image(delta_sigma.unsqueeze(0), fill_value=0, res=128)
            sigma_batched.append(delta_sigma_img)

        sigma_batched = torch.cat(sigma_batched).to(sigma.device)
        return sigma_batched

    def forward_layer(self, sigma, delta_sigma, layer_idx):
        """
        y: torch.tensor, measurements
        sigma: torch.tensor, current prediction
        delta_sigma: torch.tensor, update from Gauss-Newton based on current sigma 
        layer_idx: int, which layer to evaluate 
        
        """

        sigma_new = self.blocks[layer_idx](torch.cat([sigma, delta_sigma], dim=1))
        return sigma_new
    
    def interpolate_to_image(self, sigma, fill_value=0.0, res=256):
        coordinates = self.xy
        cells = self.cells

        pos = [
            [
                (
                    coordinates[cells[i, 0], 0]
                    + coordinates[cells[i, 1], 0]
                    + coordinates[cells[i, 2], 0]
                )
                / 3.0,
                (
                    coordinates[cells[i, 0], 1]
                    + coordinates[cells[i, 1], 1]
                    + coordinates[cells[i, 2], 1]
                )
                / 3.0,
            ]
            for i in range(cells.shape[0])
        ]
        pos = np.array(pos)

        pixcenter_x = np.linspace(np.min(pos), np.max(pos), res)
        pixcenter_y = pixcenter_x
        X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
        pixcenters = np.column_stack((X.ravel(), Y.ravel()))

        sigma_pix_ = []
        for i in range(sigma.shape[0]):
            interp = LinearNDInterpolator(
                pos, torch.clone(sigma[i, :]).cpu().numpy(), fill_value=fill_value
            )
            sigma_grid = interp(pixcenters)

            sigma_pix = np.flipud(sigma_grid.reshape(res, res))
            sigma_pix = (
                torch.from_numpy(sigma_pix.copy()).float().to(sigma.device).unsqueeze(0)
            )
            sigma_pix_.append(sigma_pix)

        return torch.stack(sigma_pix_, dim=0)


class GraphNaiveIterativeNet(nn.Module):
    def __init__(self, solver, n_iter, backCond, gn_reconstructor, n_layer=4, internal_ch=32,
                 kernel_size=3, batch_norm=True, prelu=False, lrelu_coeff=0.2):
        super(GraphNaiveIterativeNet, self).__init__()
        self.n_iter = n_iter

        self.solver = solver 
        self.backCond = backCond 
        self.gn_reconstructor = gn_reconstructor 

        self.mesh_pos = np.array(self.solver.V_sigma.tabulate_dof_coordinates()[:, :2])
        self.xy = self.solver.omega.geometry.x
        self.cells = self.solver.omega.geometry.dofmap.reshape(
        (-1, self.solver.omega.topology.dim + 1) )


        self.blocks = nn.ModuleList()
        for it in range(n_iter):
            self.blocks.append(get_unet_model(
                        in_ch=2,
                        out_ch=1,
                        scales=3,
                        skip=4,
                        channels=(32, 32, 64),
                        use_sigmoid=False,
                        use_norm=True))

            #self.blocks.append(IterativeBlock(
            #    n_in=2, n_out=1, n_layer=n_layer,
            #    internal_ch=internal_ch, kernel_size=kernel_size,
            #    batch_norm=batch_norm, prelu=prelu, lrelu_coeff=lrelu_coeff))

    def freeze_blocks(self, n):
        # Freeze parameters of all blocks up to block n
        for i in range(n + 1):
            for param in self.blocks[i].parameters():
                param.requires_grad = False

    def forward(self, Umeas):
        sigma_reco = Function(self.solver.V_sigma)
        sigma = torch.ones(1, len(sigma_reco.x.array[:]), device=Umeas.device) * self.backCond
        #sigma = torch.ones(1, 1, 128, 128, device=Umeas.device) * self.backCond
        from tqdm import tqdm 
        for i in tqdm(range(self.n_iter)):
            
            # This part is not differentiable, so we have to train greedy
            delta_sigma_img = self.compute_delta_sigma(sigma, Umeas) 
            #sigma = self.blocks[i](torch.cat([sigma, delta_sigma_img], dim=1))
            sigma = sigma + 0.2 * delta_sigma_img
        return sigma

    def compute_delta_sigma(self, sigma, Umeas):
        
        sigma_batched = []
        for i in range(sigma.shape[0]):
            delta_sigma = self.gn_reconstructor.single_step(sigma[i].flatten().cpu(), Umeas[i].cpu().numpy())
            sigma_batched.append(delta_sigma)

        sigma_batched = torch.cat(sigma_batched).to(sigma.device)
        return sigma_batched

    def forward_layer(self, sigma, delta_sigma, layer_idx):
        """
        y: torch.tensor, measurements
        sigma: torch.tensor, current prediction
        delta_sigma: torch.tensor, update from Gauss-Newton based on current sigma 
        layer_idx: int, which layer to evaluate 
        
        """

        sigma_new = self.blocks[layer_idx](torch.cat([sigma, delta_sigma], dim=1))
        return sigma_new
    

"""
There will be two different LearnedIterativeReconstructors: Image-based vs. Mesh-based

For Image-based
Input: measurements

1. One step GN-TV
2. Interpolate to Image
3. Run small UNet
4. Interpolate to Mesh

We have to train this in a greedy way. We need a functionality to save intermediate datasets.
"""
#class LearnedIterativeReconstructor(Reconstructor):
#    def __init__(self, eit_solver, device, config):
#        super().__init__(eit_solver)

#        self.device = device 
#        self.config = config 

