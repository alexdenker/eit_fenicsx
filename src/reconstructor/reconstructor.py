from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import LinearNDInterpolator


class Reconstructor(ABC):
    def __init__(self, eit_solver):
        self.eit_solver = eit_solver

    @abstractmethod
    def forward(self, U, **kwargs):
        pass

    def interpolate_to_image(self, sigma, fill_value=0):
        coordinates = self.eit_solver.omega.geometry.x
        cells = self.eit_solver.omega.geometry.dofmap.reshape(
            (-1, self.eit_solver.omega.topology.dim + 1)
        )

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

        pixcenter_x = np.linspace(np.min(pos), np.max(pos), 256)
        pixcenter_y = pixcenter_x
        X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
        pixcenters = np.column_stack((X.ravel(), Y.ravel()))

        # interp = NearestNDInterpolator(pos, sigma)
        interp = LinearNDInterpolator(pos, sigma, fill_value=fill_value)
        sigma_grid = interp(pixcenters)

        sigma_pix = np.flipud(sigma_grid.reshape(256, 256))

        return sigma_pix
