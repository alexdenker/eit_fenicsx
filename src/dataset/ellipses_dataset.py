import os

import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


class EllipsesDataset(Dataset):
    def __init__(self, part="train", base_path="dataset", inj_mode="all_against_1"):
        assert part in [
            "train",
            "val",
            "test",
            "ood"
        ], "Part has to be either train, val, test or ood"
        assert inj_mode in [
            "all_against_1",
            "all",
        ], "inj_mode has to be either all_against_1 or all"

        self.part = part
        self.base_path = base_path
        self.inj_mode = inj_mode

        self.Inj_pattern = np.load(f"{base_path}/injection_pattern.npy")
        if self.inj_mode == "all_against_1":
            # the last 15 rows are "all_against_1"
            self.Inj_pattern = self.Inj_pattern[-15:, :]

        # only for visualisation, these two arrays define the triangulation
        self.xy = np.load(f"{base_path}/mesh_points.npy")
        self.cells = np.load(f"{base_path}/cells.npy")
        self.tri = Triangulation(self.xy[:, 0], self.xy[:, 1], self.cells)

        self.sigma_files = [
            f
            for f in os.listdir(os.path.join(self.base_path, self.part))
            if f.startswith("sigma")
        ]
        self.sigma_files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

        if self.part == "test" or self.part == "ood":
            # load noisy data
            self.Umeas_files = [
                f
                for f in os.listdir(os.path.join(self.base_path, self.part))
                if f.startswith("Umeas_noisy")
            ]
            self.Umeas_files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        else:
            # load clean data
            self.Umeas_files = [
                f
                for f in os.listdir(os.path.join(self.base_path, self.part))
                if not f.startswith("Umeas_noisy") and f.startswith("Umeas")
            ]
            self.Umeas_files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))

    def __len__(self):
        return len(self.sigma_files)

    def __getitem__(self, IDX):
        sigma = np.load(os.path.join(self.base_path, self.part, self.sigma_files[IDX]))
        Umeas = np.load(os.path.join(self.base_path, self.part, self.Umeas_files[IDX]))
        if self.inj_mode == "all_against_1":
            Umeas = Umeas[-15:, :]

        sigma = torch.from_numpy(sigma).float()
        Umeas = torch.from_numpy(Umeas).float()
        return sigma, Umeas

    def visualise_phantom(self, sigma):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        im = ax.tripcolor(
            self.tri,
            sigma.cpu().numpy().flatten(),
            cmap="jet",
            shading="flat",
            vmin=0.01,
            vmax=4.0,
            edgecolors="k",
        )
        ax.axis("image")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Phantom")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return fig

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


if __name__ == "__main__":
    dataset = EllipsesDataset(part="test", inj_mode="all")

    sigma, Umeas = dataset[45]
    print(sigma.shape, Umeas.shape)

    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=32)

    s, U = next(iter(dl))

    s_pix = dataset.interpolate_to_image(s)

    print(s.shape, s_pix.shape, U.shape)

    fig = dataset.visualise_phantom(s[0, :])

    fig, ax = plt.subplots(1, 1)
    ax.imshow(
        s_pix[0, 0, :, :].cpu().numpy(),
        cmap="jet",
        vmin=0.01,
        vmax=4.0,
    )
    plt.show()
