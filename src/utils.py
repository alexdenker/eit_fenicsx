import numpy as np
from scipy.interpolate import interpn, NearestNDInterpolator


def image_to_mesh(x, mesh_pos, fill_value=1.0):
    """
    Interpolate image x to mesh given by mesh positions. 

    x: [1, H, W] numpy array 
    
    """
    assert len(x.shape) == 3, f"wrong shape of image: {x.shape}"

    radius = np.max(np.abs(mesh_pos))

    pixcenter_x = pixcenter_y = np.linspace(-radius, radius, x.shape[-1])
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")
    sigma = interpn(
        [pixcenter_x, pixcenter_y],
        np.flipud(x[0]).T, 
        mesh_pos,
        bounds_error=False,
        fill_value=fill_value,
        method="nearest",
    )
    
    return sigma


def interpolate_mesh_to_mesh(x, mesh_pos1, mesh_pos2):
    interpolator = NearestNDInterpolator(mesh_pos1, x)

    sigma = interpolator(mesh_pos2[:, 0], mesh_pos2[:, 1])

    return sigma

