import numpy as np
import os
import time
import sys

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import torch
from omegaconf import OmegaConf

from dolfinx.fem import Function

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eit_forward_fenicsx import EIT
from src.sparsity_reconstruction import L1Sparsity
from src.gauss_newton import GaussNewtonSolverTV, LinearisedReconstruction
from src.data_driven_reconstructors import FCUnet
from src.random_ellipses import gen_conductivity
from src.performance_metrics import (
    RelativeL1Error,
    DiceScore,
    DynamicRange,
    MeasurementError,
    RelativeL2Error
)

from ellipses_dataset import EllipsesDataset
from kit4_dataset import KIT4Dataset
import time

import argparse

parser = argparse.ArgumentParser(description="conditional sampling")

parser.add_argument("--method", default="l1_sparsity")
parser.add_argument("--dataset", default="ellipses")  # ellipses or kit4
parser.add_argument("--part", default="val")

# used for l1 sparsity
parser.add_argument("--alpha", default=0.0001)
parser.add_argument("--kappa", default=0.0285)


# used for gauss-newton with TV
parser.add_argument("--lamb", default=0.04)
parser.add_argument("--beta", default=1e-6)
parser.add_argument("--num_steps", default=8)


def main(args):
    part = str(args.part)
    inj_mode = "all"
    delta = 0.005  # noise level
    method = args.method

    if method == "l1_sparsity":
        from config.l1_sparsity import get_config

        config = get_config()

        config.alpha = float(args.alpha)
        config.kappa = float(args.kappa)
    elif method == "gn_tv":
        from config.gn_tv import get_config

        config = get_config()

        config.lamb = float(args.lamb)
        config.beta = float(args.beta)
        config.num_steps = int(args.num_steps)
    elif method == "fcunet":
        from config.fcunet import get_config

        config = get_config()
    elif method == "linearised_reco":
        from config.linearised_reco import get_config

        config = get_config()
        config.lamb = float(args.lamb)
    else:
        raise NotImplementedError

    L = 16
    backCond = 1.31

    config.backCond = backCond
    config.part = part
    config.inj_mode = inj_mode
    config.delta = delta

    print(config)
    save_dir = os.path.join(
        "results",
        str(args.dataset),
        method,
        part,
        f'{time.strftime("%d-%m-%Y-%H-%M-%S")}',
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "imgs")):
        os.makedirs(os.path.join(save_dir, "imgs"))

    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        OmegaConf.save(config, file)

    if str(args.dataset) == "ellipses":
        dataset = EllipsesDataset(part="test", inj_mode=inj_mode)
        z = 1e-6 * np.ones(L)

        if part == "val":
            max_idx = min(10, len(dataset))
        else:
            max_idx = len(dataset)  # min(10, len(dataset))
    elif str(args.dataset) == "kit4":
        dataset = KIT4Dataset(inj_mode=inj_mode)
        z = np.array(
            [
                0.00880276,
                0.00938687,
                0.00989395,
                0.01039582,
                0.00948009,
                0.00943006,
                0.01016697,
                0.0088116,
                0.00802456,
                0.0090383,
                0.00907472,
                0.00847228,
                0.00814984,
                0.00877861,
                0.00841414,
                0.00877331,
            ]
        )

        max_idx = len(dataset)
    else:
        raise NotImplementedError

    print("dataset mode: ", dataset.inj_mode)

    Injref = dataset.Inj_pattern

    solver = EIT(L, Injref, z, backend="Scipy", mesh_name="data/KIT4_mesh_coarse.msh")

    xy = solver.omega.geometry.x
    cells = solver.omega.geometry.dofmap.reshape((-1, solver.omega.topology.dim + 1))
    tri = Triangulation(xy[:, 0], xy[:, 1], cells)

    sigma_gt_vsigma = Function(solver.V_sigma)
    # print("Len: ", len(sigma_gt_vsigma.x.array[:]))

    if method == "l1_sparsity":
        reconstructor = L1Sparsity(
            eit_solver=solver,
            backCond=config.backCond,
            kappa=config.kappa,
            clip=[config.l1, config.l2],
            max_iter=config.max_iter,
            stopping_criterion=config.stopping_criterion,
            step_min=config.step_min,
            initial_step_size=config.initial_step_size,
            alpha=config.alpha,
        )
    elif method == "gn_tv":
        reconstructor = GaussNewtonSolverTV(
            eit_solver=solver,
            device="cuda",
            num_steps=config.num_steps,
            lamb=config.lamb,
            beta=config.beta,
            clip=[config.l1, config.l2],
        )
    elif method == "fcunet":
        reconstructor = FCUnet(
            eit_solver=solver, device="cuda", load_path=config.load_path
        )
        reconstructor.model.eval()

        print(
            "Number of parameters: ",
            sum([p.numel() for p in reconstructor.model.parameters()]),
        )
    elif method == "linearised_reco":
        if isinstance(dataset, KIT4Dataset):
            Uel_background = dataset.Uel_background
        else:
            sigma_background = Function(solver.V_sigma)
            sigma_background.x.array[:] = config.backCond

            _, Uel_background = solver.forward_solve(sigma_background)
            Uel_background = np.array(Uel_background).flatten()

        Lprior = np.load("data/L_KIT4_mesh_coarse.npy")
        Lprior = torch.from_numpy(Lprior).float()

        reconstructor = LinearisedReconstruction(
            eit_solver=solver,
            device="cuda",
            R=Lprior.T @ Lprior,
            lamb=config.lamb,
            Uel_background=Uel_background,
            clip=[config.l1, config.l2],
            backCond=config.backCond,
        )
    else:
        raise NotImplementedError

    results_dict = {}

    rel_error_l1_list = []
    rel_error_l2_list = []
    dice_score_list = []
    inf_time_list = []
    dr_list = []
    measurement_error_list = []

    rel_l1_error = RelativeL1Error(name="RelL1")
    rel_l2_error = RelativeL2Error(name="RelL2")
    dynamic_range = DynamicRange(name="DR")
    dice_score = DiceScore(name="Dice", backCond=config.backCond)
    measurement_error = MeasurementError(name="VoltageError", solver=solver)

    for idx in range(max_idx):
        if isinstance(dataset, EllipsesDataset):
            s, U = dataset[idx]
            sigma_gt_vsigma = Function(solver.V_sigma)
            sigma_gt_vsigma.x.array[:] = s.cpu().numpy()

            _, U = solver.forward_solve(sigma_gt_vsigma)

        elif isinstance(dataset, KIT4Dataset):
            s, U = dataset[idx]
        else:
            raise NotImplementedError

        # Umeas = U.numpy()
        Umeas = np.array(U)
        if part == "val" and isinstance(dataset, EllipsesDataset):
            # add noise to validation data
            Umeas = Umeas + delta * np.mean(np.abs(Umeas)) * np.random.normal(
                size=Umeas.shape
            )

        if method == "gn_tv" or method == "linearised_reco":
            noise_percentage = delta  # 0.01
            var_meas = (noise_percentage * np.abs(Umeas)) ** 2
            GammaInv = 1.0 / (var_meas.flatten() + 0.001)
            GammaInv = torch.from_numpy(GammaInv).float().to(reconstructor.device)

            reconstructor.GammaInv = GammaInv

        sigma_init = Function(solver.V_sigma)
        sigma_init.x.array[:] = backCond

        t_start = time.time()
        sigma_reco = reconstructor.forward(
            Umeas=Umeas, verbose=True, sigma_init=sigma_init
        )


        if isinstance(reconstructor, L1Sparsity):
            sigma_reco_l1_vsigma = Function(solver.V_sigma)
            sigma_reco_l1_vsigma.interpolate(sigma_reco)
            sigma_reco = sigma_reco_l1_vsigma

        t_end = time.time()
        inf_time_list.append(t_end - t_start)
        if isinstance(dataset, EllipsesDataset):
            # Calculate Quality Metrics
            dr_list.append(dynamic_range(sigma_reco, sigma_gt_vsigma))
            dice_score_list.append(dice_score(sigma_reco, sigma_gt_vsigma))
            measurement_error_list.append(measurement_error(sigma_reco, Umeas))
            rel_error_l1_list.append(rel_l1_error(sigma_reco, sigma_gt_vsigma))
            rel_error_l2_list.append(rel_l2_error(sigma_reco, sigma_gt_vsigma))

            sigma_img = reconstructor.interpolate_to_image(
                np.array(sigma_reco.x.array[:]).flatten(), fill_value=backCond
            )
            sigma_img_gt = reconstructor.interpolate_to_image(
                np.array(sigma_gt_vsigma.x.array[:]).flatten(), fill_value=backCond
            )

            fig, axes = plt.subplots(2, 2, figsize=(12, 6))

            pred = np.array(sigma_gt_vsigma.x.array[:]).flatten()
            im = axes[0, 0].tripcolor(
                tri,
                pred,
                cmap="jet",
                shading="flat",
                vmin=0.01,
                vmax=3.0,
                edgecolor="k",
            )
            axes[0, 0].axis("image")
            axes[0, 0].set_aspect("equal", adjustable="box")
            axes[0, 0].set_title("GT")
            fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
            axes[0, 0].axis("off")

            pred = np.array(sigma_reco.x.array[:]).flatten()
            im = axes[0, 1].tripcolor(
                tri,
                pred,
                cmap="jet",
                shading="flat",
                vmin=0.01,
                vmax=3.0,
                edgecolor="k",
            )
            axes[0, 1].axis("image")
            axes[0, 1].set_aspect("equal", adjustable="box")
            axes[0, 1].set_title(
                f"Reconstruction, \n rel. L1 error={np.format_float_positional(rel_error_l1_list[-1],4)}"
            )
            fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
            axes[0, 1].axis("off")

            im = axes[1, 0].imshow(sigma_img_gt, cmap="jet", vmin=0.01, vmax=3.0)
            axes[1, 0].set_title("GT")
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
            axes[1, 0].axis("off")

            im = axes[1, 1].imshow(sigma_img, cmap="jet", vmin=0.01, vmax=3.0)
            axes[1, 1].set_title(
                f"Reconstruction, \n rel. L1 error={np.format_float_positional(rel_error_l1_list[-1],4)}"
            )
            fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
            axes[1, 1].axis("off")

            plt.savefig(os.path.join(save_dir, "imgs", f"img_{idx}.png"))
            # plt.show()
            plt.close()

        elif isinstance(dataset, KIT4Dataset):
            sigma_img = reconstructor.interpolate_to_image(
                np.array(sigma_reco.x.array[:]).flatten(), fill_value=backCond
            )

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

            ax1.imshow(s.permute(1, 2, 0).cpu().numpy())
            ax1.axis("off")

            ax3.imshow(sigma_img, cmap="jet", vmin=0.01, vmax=3.0)
            ax3.axis("off")

            pred = np.array(sigma_reco.x.array[:]).flatten()
            im = ax2.tripcolor(
                tri,
                pred,
                cmap="jet",
                shading="flat",
                edgecolor="k",
                vmin=0.01,
                vmax=3.0,
            )
            ax2.axis("image")
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_title("Gauss-Newton with TV")
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            ax2.axis("off")

            plt.savefig(os.path.join(save_dir, "imgs", f"img_{idx}.png"))
            plt.close()

    results_dict["rel_l1_error"] = float(np.mean(rel_error_l1_list))
    results_dict["rel_l2_error"] = float(np.mean(rel_error_l2_list))
    results_dict["inf_time"] = float(np.mean(inf_time_list))
    results_dict["dice_score"] = float(np.mean(dice_score_list))
    results_dict["dynamic_range"] = float(np.mean(dr_list))
    results_dict["measurement_error"] = float(np.mean(measurement_error_list))

    with open(os.path.join(save_dir, "results.yaml"), "w") as file:
        yaml.dump(results_dict, file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
