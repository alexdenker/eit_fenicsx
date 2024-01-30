"""
Written by KTC2023 challenge organizers:
https://zenodo.org/record/8252370

"""

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

class SigmaPlotter:
    def __init__(self, Mesh, fh, cmp):
        self.Mesh = Mesh
        self.cmp = cmp
        handle1 = plt.figure(fh[0])
        handle1.set_size_inches(8, 6)
        
        if len(fh) == 2:
            handle2 = plt.figure(fh[1])
            handle2.set_size_inches(8, 6)
            self.fh = [handle1, handle2]
        else:
            self.fh = [handle1]

    def basicplot(self, sigma, str):
        self.basic2Dplot(sigma, str)

    def basic2Dplot(self, sigma, delta_sigma, str):
        plt.figure(self.fh[0].number)
        plt.clf()
        self.plot_solution(self.Mesh.g, self.Mesh.H, sigma, self.fh[0])
        plt.colorbar()
        plt.axis('image')
        plt.set_cmap(self.cmp)
        plt.ion()
        plt.show()
        if str is not None:
            plt.title(str[0])

        if len(self.fh) == 2:
            plt.figure(self.fh[1].number)
            plt.clf()
            self.plot_solution(self.Mesh.g, self.Mesh.H, delta_sigma, self.fh[1])
            plt.colorbar()
            plt.axis('image')
            plt.set_cmap(self.cmp)
            if str is not None:
                plt.title(str[1])
        
        plt.draw()
        plt.pause(0.001)
        
        
    @staticmethod
    def plot_solution(g, H, s, fighandle=None):
        if fighandle is not None:
            plt.figure(fighandle.number)

        if g.shape[1] < 3:
            z = s
        else:
            z = s

        tri = Triangulation(g[:, 0], g[:, 1], H)
        plt.tripcolor(tri, np.array(z).flatten(), shading='gouraud', cmap=plt.get_cmap())#
        plt.axis('image')
        plt.gca().set_aspect('equal', adjustable='box')
