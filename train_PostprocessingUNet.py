import os
import yaml

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from dolfinx.fem import Function

from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from src import EllipsesDataset,  LinearisedReconstruction, EIT, AttentionUNetModel

config = {
    "inj_mode": "all",
    "training": {
        "batch_size": 12,
        "epochs": 200,
        "lr": 1e-4,
    },
    "model": {
        "image_size": 256,
        "in_channels": 1,
        "model_channels": 32,
        "out_channels": 1,
        "channel_mult": (1, 2, 2, 4, 4),
        "num_res_blocks": 2,
        "attention_resolutions": ("32", "16"),
    },
    "linearised_reco": {
        "lamb": 8.04,
        "l1":  0.01,
        "l2": 3.0,
    }
}

log_dir = "models/PostUNet"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

found_version = False
version_num = 1
while not found_version:
    if os.path.isdir(os.path.join(log_dir, "version_{:02d}".format(version_num))):
        version_num += 1
    else:
        found_version = True

log_dir = os.path.join(log_dir, "version_{:02d}".format(version_num))
print("save model to ", log_dir)
os.makedirs(log_dir)

with open(os.path.join(log_dir, "report.yaml"), "w") as file:
    yaml.dump(config, file)

train_dataset_base = EllipsesDataset(part="train", inj_mode=config["inj_mode"])
#train_dataset_base_subset = torch.utils.data.Subset(train_dataset_base, np.arange(24))
val_dataset_base = EllipsesDataset(part="val", inj_mode=config["inj_mode"])
#val_dataset_base_subset = torch.utils.data.Subset(val_dataset_base, np.arange(24))

Injref = train_dataset_base.Inj_pattern

L = 16
z = 1e-6 * np.ones(L)
backCond = 1.31
solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")

sigma_background = Function(solver.V_sigma)
sigma_background.x.array[:] = backCond

_, Uel_background = solver.forward_solve(sigma_background)
Uel_background = np.array(Uel_background).flatten()

if not os.path.exists("data/L_KIT4_mesh_coarse.npy"):
    from src import create_smoothness_regulariser
    create_smoothness_regulariser(
    solver.omega, "data/L_KIT4_mesh_coarse.npy", corrlength=0.2, std=0.15
    )

Lprior = np.load("data/L_KIT4_mesh_coarse.npy")
Lprior = torch.from_numpy(Lprior).float()

reconstructor = LinearisedReconstruction(
    eit_solver=solver,
            device="cuda",
            R=Lprior.T @ Lprior,
            Uel_background=Uel_background,
            backCond=backCond,
            lamb=config["linearised_reco"]["lamb"],
            clip=[config["linearised_reco"]["l1"], config["linearised_reco"]["l2"]],
)

def save_transformed_data(data_loader,reconstructor, output_dir, part):
    print("Save intermediate data")
    output_file = os.path.join(output_dir, f"{part}_input_data.pt")

    transformed_data = []
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        sigma, Umeas = batch

        delta1 = 0.1  # noise level
        delta2 = 0.001 
        var_meas = (delta1 * np.abs(Umeas.flatten().numpy() - Uel_background) + delta2 * np.max(np.abs(Umeas.flatten().numpy() - Uel_background))) ** 2
        GammaInv = 1.0 / (np.maximum(var_meas.flatten(),1e-5))
        GammaInv = torch.from_numpy(GammaInv).float().to(reconstructor.device)

        reconstructor.GammaInv = GammaInv

        sigma_init = Function(solver.V_sigma)
        sigma_init.x.array[:] = backCond

        sigma_reco = reconstructor.forward(
            Umeas=Umeas.numpy(), verbose=False, sigma_init=sigma_init
        )
        sigma_img = reconstructor.interpolate_to_image(
                np.array(sigma_reco.x.array[:]).flatten(), fill_value=backCond
            )
        sigma_img_gt = reconstructor.interpolate_to_image(
                sigma.cpu().flatten(), fill_value=backCond
            )

        transformed_data.append((torch.from_numpy(sigma_img_gt.copy()).float().unsqueeze(0), torch.from_numpy(sigma_img.copy()).float().unsqueeze(0)))

    torch.save(transformed_data, output_file)
    print(f"Transformed data {output_file}")
    return output_file

class IntermediateDataset(torch.utils.data.Dataset):
    def __init__(self, path):

        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, IDX):
        sigma, sigma_reco = self.data[IDX]

        return sigma, sigma_reco


model = AttentionUNetModel(
    image_size=config["model"]["image_size"],
                in_channels=config["model"]["in_channels"],
                model_channels=config["model"]["model_channels"],
                out_channels=config["model"]["out_channels"],
                num_res_blocks=config["model"]["num_res_blocks"],
                channel_mult=config["model"]["channel_mult"],
                attention_resolutions=config["model"]["attention_resolutions"],
)
model.to("cuda")

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

log_every = 50
save_every = 10
writer = SummaryWriter(log_dir=log_dir, comment="training-fc-unet")

optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

scheduler = lr_scheduler.CosineAnnealingLR(
optimizer,
T_max=config["training"]["epochs"],
eta_min=config["training"]["lr"] / 10.0,
)

best_val = 1e8

save_data_loader = DataLoader(train_dataset_base, batch_size=1, shuffle=False)
train_data_name = save_transformed_data(save_data_loader,reconstructor, output_dir=log_dir, part="train")
save_data_loader = DataLoader(val_dataset_base, batch_size=1, shuffle=False)
val_data_name = save_transformed_data(save_data_loader, reconstructor, output_dir=log_dir, part="val") 

dataset_train_block = IntermediateDataset(os.path.join(log_dir, train_data_name))
dataset_val_block = IntermediateDataset(os.path.join(log_dir, val_data_name))

train_loader = DataLoader(dataset_train_block, batch_size=config["training"]["batch_size"], shuffle=True)
val_loader = DataLoader(dataset_val_block, batch_size=config["training"]["batch_size"], shuffle=False)

for epoch in range(config["training"]["epochs"]):
    model.train()
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        sigma, sigma_reco = batch

        sigma = sigma.to("cuda")
        sigma_reco = sigma_reco.to("cuda")

        x_pred = model(sigma_reco)
        loss = torch.mean((x_pred - sigma) ** 2)

        loss.backward()

        optimizer.step()

        if idx % log_every == 0:
            writer.add_scalar("train/loss", loss.item(), epoch * len(train_loader) + idx)

    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Current learning rate: ", after_lr)
    writer.add_scalar("learning_rate", after_lr, epoch + 1)
    if (epoch + 1) % save_every == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))

    model.eval()

    val_loss = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            sigma, sigma_reco = batch

            sigma = sigma.to("cuda")
            sigma_reco = sigma_reco.to("cuda")

            x_pred = model(sigma_reco)
            loss = torch.mean((x_pred - sigma) ** 2)

            val_loss.append(loss.item())

            if idx == 0:
                for i in range(sigma.shape[0]):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

                    ax1.imshow(
                        sigma[i, 0, :, :].cpu().numpy(),
                        cmap="jet",
                        vmin=0.01,
                        vmax=4.0,
                    )
                    ax1.set_title(f"Ground truth for {i}")
                    ax1.axis("off")

                    ax2.imshow(
                        x_pred[i, 0, :, :].detach().cpu().numpy(),
                        cmap="jet",
                        vmin=0.01,
                        vmax=4.0,
                    )
                    ax2.set_title(f"Prediction for {i}")
                    ax2.axis("off")

                    ax3.imshow(
                        sigma_reco[i, 0, :, :].detach().cpu().numpy(),
                        cmap="jet",
                        vmin=0.01,
                        vmax=4.0,
                    )
                    ax3.set_title(f"UNet input for {i}")
                    ax3.axis("off")

                    writer.add_figure(f"val/image_{i}", fig, global_step=epoch)
                    plt.close()

    mean_val_loss = np.mean(val_loss)
    writer.add_scalar("val/loss", mean_val_loss, epoch)

    if mean_val_loss < best_val:
        print("New best score: ", mean_val_loss, " previous: ", best_val)
        best_val = mean_val_loss

        torch.save(model.state_dict(), os.path.join(log_dir, "model_bestval.pt"))

    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))



