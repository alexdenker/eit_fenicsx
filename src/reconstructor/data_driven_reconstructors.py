import os
import yaml
import torch
import numpy as np
from dolfinx.fem import Function

from src.reconstructor import Reconstructor
from src.networks import get_fcunet_model, FNO_dse, AttentionUNetModel, FCAttentionUNetModel
from src.utils import image_to_mesh

class FCUnet(Reconstructor):
    def __init__(self, eit_solver, device, load_path):
        super().__init__(eit_solver)

        with open(os.path.join(load_path, "report.yaml"), "r") as file:
            config = yaml.load(file, Loader=yaml.UnsafeLoader)
        print("config: ", config)
        if config["inj_mode"] == "all":
            in_dim = 1264
        elif config["inj_mode"] == "all_against_1":
            in_dim = 240
        else:
            raise NotImplementedError

        self.device = device

        self.model = FCAttentionUNetModel(
            in_dim=in_dim,
            in_res=config["model"]["in_res"],
            image_size=config["model"]["image_size"],
            in_channels=config["model"]["in_channels"],
            model_channels=config["model"]["model_channels"],
            out_channels=config["model"]["out_channels"],
            num_res_blocks=config["model"]["num_res_blocks"],
            channel_mult=config["model"]["channel_mult"],
            attention_resolutions=config["model"]["attention_resolutions"],
        )
        self.model.load_state_dict(
            torch.load(os.path.join(load_path, "best_val_model.pt"))
        )
        self.model.to(device)
        self.model.eval()

    def forward(self, Umeas, **kwargs):
        self.model.eval()

        Umeas = torch.from_numpy(Umeas.flatten()).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            sigma_pred = self.model(Umeas).squeeze().cpu().numpy()

        # interpolates sigma_pred to mesh
        mesh_pos = np.array(self.eit_solver.V_sigma.tabulate_dof_coordinates()[:, :2])
        sigma = image_to_mesh(sigma_pred, mesh_pos=mesh_pos)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco


class FCFNONet(Reconstructor):
    def __init__(self, eit_solver, device, load_path):
        super().__init__(eit_solver)

        with open(os.path.join(load_path, "report.yaml"), "r") as file:
            config = yaml.load(file, Loader=yaml.UnsafeLoader)
        print("config: ", config)
        if config["inj_mode"] == "all":
            in_dim = 1264
        elif config["inj_mode"] == "all_against_1":
            in_dim = 240
        else:
            raise NotImplementedError

        self.device = device

        out_dim = 4728
        self.initial_layer = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.initial_layer.load_state_dict(
            torch.load(os.path.join(load_path, "best_val_initial_linear.pt"))
        )
        self.initial_layer.to(device)

        self.model = FNO_dse(modes=config["model"]["modes"], width=config["model"]["width"], num_blocks=config["model"]["num_blocks"], use_batch_norm=config["model"]["use_batch_norm"])
        self.model.load_state_dict(
            torch.load(os.path.join(load_path, "best_val_model.pt"))
        )
        self.model.to(device)
        self.model.eval()

    def forward(self, Umeas, **kwargs):
        self.model.eval()

        Umeas = torch.from_numpy(Umeas.flatten()).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            sigma_init = self.initial_layer(Umeas)

            pos = torch.from_numpy(np.array(self.eit_solver.V_sigma.tabulate_dof_coordinates()[:, :2])).float().to(self.device)
            pos_inp = torch.repeat_interleave(pos.unsqueeze(0), sigma_init.shape[0], dim=0).to(self.device)
            sigma_init = sigma_init.unsqueeze(-1)
            model_inp = torch.cat([pos_inp, sigma_init], dim=-1)
            sigma_pred = self.model(model_inp)[:,:,0].squeeze().cpu().numpy()

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma_pred.flatten()
        return sigma_reco
    

class PostprocessingUNet(Reconstructor):
    def __init__(self, eit_solver, reconstructor, device, load_path):
        super().__init__(eit_solver)

        with open(os.path.join(load_path, "report.yaml"), "r") as file:
            config = yaml.load(file, Loader=yaml.UnsafeLoader)
        print("config: ", config)


        self.device = device

        self.reconstructor = reconstructor

        self.model = AttentionUNetModel(
                image_size=config["model"]["image_size"],
                in_channels=config["model"]["in_channels"],
                model_channels=config["model"]["model_channels"],
                out_channels=config["model"]["out_channels"],
                num_res_blocks=config["model"]["num_res_blocks"],
                channel_mult=config["model"]["channel_mult"],
                attention_resolutions=config["model"]["attention_resolutions"],
        )
        self.model.load_state_dict(
            torch.load(os.path.join(load_path, "model_bestval.pt"))
        )
        self.model.to(device)
        self.model.eval()

    def forward(self, Umeas, **kwargs):
        self.model.eval()

        backCond = kwargs.get("backCond", 1.31)

        Uel_background = self.reconstructor.Uel_background
        if Uel_background is None:
            print("No background conductivity is specified")
            Uel_background = 0 

        delta1 = 0.1  # noise level
        delta2 = 0.001 
        var_meas = (delta1 * np.abs(Umeas.flatten() - Uel_background) + delta2 * np.max(np.abs(Umeas.flatten() - Uel_background))) ** 2
        GammaInv = 1.0 / (np.maximum(var_meas.flatten(),1e-5))
        GammaInv = torch.from_numpy(GammaInv).float().to(self.device)

        self.reconstructor.GammaInv = GammaInv

        sigma_init = Function(self.eit_solver.V_sigma)
        sigma_init.x.array[:] = backCond

        sigma_reco = self.reconstructor.forward(
            Umeas=Umeas, verbose=False, sigma_init=sigma_init
        )
        sigma_img = self.reconstructor.interpolate_to_image(
                np.array(sigma_reco.x.array[:]).flatten(), fill_value=backCond
            )
        sigma_img = torch.from_numpy(sigma_img.copy()).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sigma_pred = self.model(sigma_img).squeeze().cpu().numpy()

        # interpolates sigma_pred to mesh
        mesh_pos = np.array(self.eit_solver.V_sigma.tabulate_dof_coordinates()[:, :2])
        sigma = image_to_mesh(sigma_pred, mesh_pos=mesh_pos)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco
