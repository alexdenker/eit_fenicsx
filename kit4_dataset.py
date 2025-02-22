import os

import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.io import loadmat


from PIL import Image
import numpy as np


class KIT4Dataset(Dataset):
    def __init__(
        self,
        base_path="KIT4/data_mat_files",
        inj_mode="all_against_1",
        img_path="KIT4/target_photos",
    ):
        """
        Dataset to load the KIT4 data. The measurements in KIT4 is the current of adjacent electrodes.
        We instead work with the potential at single electrodes. Due to the zero potential condition (sum U_i = 0),
        there is a unique reconstruction from pairwise measurements to single measurements.

        base_path: str, path to mat files, assumed to be called "datamat_i_j.mat"
        """
        assert inj_mode in [
            "all_against_1",
            "all",
        ], "inj_mode has to be either all_against_1 or all"

        self.base_path = base_path
        self.img_path = img_path
        self.inj_mode = inj_mode

        data = loadmat(os.path.join(self.base_path, "datamat_1_0.mat"))

        B = data["MeasPattern"].T
        self.Bf = np.vstack([B, np.ones(B.shape[-1])])

        self.Inj_pattern = data["CurrentPattern"].T
        if self.inj_mode == "all_against_1":
            # the last 15 rows are "all_against_1"
            self.Inj_pattern = self.Inj_pattern[-15:, :]

        self.files = os.listdir(self.base_path)
        self.photos = os.listdir(self.img_path)

        def sort_fun(x):
            x = x.split(".")[0]

            exp = int(x.split("_")[1])
            obj = int(x.split("_")[2])

            return exp * 10 + obj

        self.files.sort(key=sort_fun)
        self.photos.sort(key=sort_fun)

        self.Uel_background = data["Uel"].T
        if self.inj_mode == "all_against_1":
            self.Uel_background = self.Uel_background[-15:, :]

        # self.sigma_files = [f for f in os.listdir(os.path.join(self.base_path, self.part)) if f.startswith("sigma")]
        # self.sigma_files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, IDX):
        data = loadmat(os.path.join(self.base_path, self.files[IDX]))

        Uel = data["Uel"].T
        if self.inj_mode == "all_against_1":
            Uel = Uel[-15:, :]

        U = []
        for i in range(Uel.shape[0]):
            U_sol, res, _, _ = np.linalg.lstsq(
                self.Bf, np.hstack([Uel[i, :], np.array([0])]), rcond=None
            )
            U.append(U_sol)

        Uel = np.stack(U)

        Umeas = torch.from_numpy(Uel).float()

        im_frame = Image.open(os.path.join(self.img_path, self.photos[IDX]))
        np_frame = np.array(im_frame) / 255.0
        im = torch.from_numpy(np_frame).float()

        return im.permute(2, 0, 1), Umeas


if __name__ == "__main__":
    dataset = KIT4Dataset()
    print(len(dataset))

    x, U = dataset[5]
    print(x.shape, U.shape)
    """
    sigma, Umeas = dataset[45]
    print(sigma.shape, Umeas.shape)

    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=32)

    s, U = next(iter(dl))
    print(s.shape, U.shape)

    fig = dataset.visualise_phantom(sigma)

    plt.show()
    """
