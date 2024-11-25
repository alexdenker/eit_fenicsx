import numpy as np 
from scipy.interpolate import interpn, NearestNDInterpolator



def image_to_mesh(x, mesh_pos):
    radius = np.max(np.abs(mesh_pos))

    pixcenter_x = pixcenter_y = np.linspace(-radius, radius, 256)
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")

    sigma = interpn([pixcenter_x, pixcenter_y], x, mesh_pos, 
        bounds_error=False, fill_value=1.0, method="nearest")

    return sigma

def interpolate_mesh_to_mesh(x, mesh_pos1, mesh_pos2):
    interpolator = NearestNDInterpolator(mesh_pos1, x)

    sigma = interpolator(mesh_pos2[:,0], mesh_pos2[:,1])


    return sigma