
import os 
import yaml 
import torch 
import numpy as np 
from dolfinx.fem import Function

from src.reconstructor import Reconstructor
from src.unet import get_fcunet_model
from src.utils import image_to_mesh

class FCUnet(Reconstructor):
    def __init__(self, 
                 eit_solver,
                 device,
                 load_path):
        super().__init__(eit_solver) 

        with open(os.path.join(load_path,'report.yaml'), 'r') as file:
            config = yaml.load(file, Loader=yaml.UnsafeLoader)
        print("config: ", config)
        if config["inj_mode"] == "all":
            in_dim = 1264
        else:
            raise NotImplementedError

        self.device = device

        self.model = get_fcunet_model(in_dim=in_dim, 
                        in_res=config["model"]["in_res"], 
                        in_ch=config["model"]["in_ch"], 
                        out_ch=config["model"]["out_ch"], 
                        scales=config["model"]["scales"], 
                        skip=config["model"]["skip"],
                        channels=config["model"]["channels"], 
                        use_sigmoid=config["model"]["use_sigmoid"],
                        use_norm=config["model"]["use_norm"])
        self.model.load_state_dict(torch.load(os.path.join(load_path, "best_val_model.pt")))
        self.model.to(device)
        self.model.eval() 

    def forward(self, Umeas, **kwargs):
        self.model.eval() 

        Umeas = torch.from_numpy(Umeas.flatten()).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            sigma_pred = self.model(Umeas).squeeze().cpu().numpy()

        # interpolates sigma_pred to mesh 
        mesh_pos = np.array(self.eit_solver.V_sigma.tabulate_dof_coordinates()[:,:2])
        sigma = image_to_mesh(sigma_pred, mesh_pos=mesh_pos)

        sigma_reco = Function(self.eit_solver.V_sigma)
        sigma_reco.x.array[:] = sigma.flatten()
        return sigma_reco