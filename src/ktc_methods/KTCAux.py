"""
Written by KTC2023 challenge organizers:
https://zenodo.org/record/8252370

"""

import numpy as np
from scipy.spatial import Delaunay

def Interpolate2Newmesh2DNode(g, H, Node, f, pts, INTPMAT):

    if INTPMAT == []:
        TR = Delaunay(g)
        Hdel = TR.simplices
        invX = [np.linalg.inv(np.column_stack((g[pp, :], np.ones(3)))) for pp in Hdel]
        np_pts = len(pts)
        Ic = np.zeros((np_pts, 3))
        Iv = np.zeros((np_pts, 3))
        Element = TR.find_simplex(pts)
        nans = np.zeros(np_pts)
        for k in range(np_pts):
            tin = Element[k]
            Phi = np.zeros(3)
            if not np.isnan(tin):
                if tin >= 0:
                    iXt = invX[tin]
                    for gin in range(3):
                        Phi[gin] = np.dot(np.append(pts[k, :], 1), iXt[:, gin])
                    Ic[k, :] = Hdel[tin, :]
                else:
                    Ic[k, :] = 1
                    nans[k] = 1
                    Iv[k, :] = 1
                Iv[k, :] = Phi
            else:
                Ic[k, :] = 1
                nans[k] = 1
                Iv[k, :] = 1
        INTPMAT = np.zeros((np_pts, f.size))
        for row in range(np_pts):
            INTPMAT[row, Ic[row].astype(int)] = Iv[row]
        INTPMAT[nans == 1, :] = 0

    f_newgrid = np.dot(INTPMAT, f)
    return f_newgrid, INTPMAT, Element

def setMeasurementPattern(Nel):
    Inj = np.matrix(np.eye(Nel))
    gnd = np.matrix(np.eye(Nel,k=-1))
    Mpat = np.matrix(Inj[:, :Nel-1] - gnd[:, :Nel-1])
    vincl = np.ones(Nel * (Nel - 1), dtype=bool)
    return Inj, Mpat, vincl

def simulateConductivity(Meshsim, inclusiontypes):
    sigma = np.ones((Meshsim.g.shape[0], 1))
    contrast = -0.5
    cp = np.array([0.5*0.115, 0.5*0.115])
    r = 0.2*0.115
    ind = np.where(np.linalg.norm(Meshsim.g.T - cp[:, None], axis=0) <= r)[0]
    delta_sigma = np.zeros_like(sigma)
    delta_sigma[ind] = contrast
    cp = np.array([0, 0])
    r = 0.2*0.115
    ind = np.where(np.linalg.norm(Meshsim.g.T - cp[:, None], axis=0) <= r)[0]
    if inclusiontypes == 2:
        delta_sigma[ind] = contrast
    else:
        delta_sigma[ind] = abs(contrast)
    sigma2 = sigma + delta_sigma
    return sigma, delta_sigma, sigma2

def interpolateRecoToPixGrid(deltareco, Mesh):
    pixwidth = 0.23 / 256
    # pixcenter_x = np.arange(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, pixwidth)
    pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
    pixcenters = np.column_stack((X.ravel(), Y.ravel()))
    deltareco_pixgrid = Interpolate2Newmesh2DNode(Mesh.g, Mesh.H, Mesh.Node, deltareco, pixcenters, [])
    deltareco_pixgrid = deltareco_pixgrid[0]
    deltareco_pixgrid = np.flipud(deltareco_pixgrid.reshape(256, 256))
    return deltareco_pixgrid

