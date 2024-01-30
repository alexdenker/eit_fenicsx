"""
Written by KTC2023 challenge organizers:
https://zenodo.org/record/8252370

"""

import numpy as np
import os

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
#   The function create2Dmesh_circ can be used as a basis to make your meshing codes using Gmsh.
#   It does not, however, run by itself, and you need to implement a way to load the Gmsh output
#   into python.

#    Mesh2 = {'H': H2, 'g': g2, 'elfaces': elfaces2, 'Node': Node2, 'Element': Element2}
#    Mesh = {'H': H, 'g': g, 'elfaces': elfaces, 'Node': Node, 'Element': Element}
#
class ELEMENT:
    def __init__(self, t, e):
        self.Topology = t
        self.Electrode = e

class NODE:
    def __init__(self, c, e):
        self.Coordinate = c
        self.ElementConnection = e

class Mesh:
    def __init__(self, H, g, elfaces, Node, Element):
        self.H = H
        self.g = g
        self.elfaces = elfaces
        self.Node = Node
        self.Element = Element

def make_node_3d_small_fast(H, g):
    rg, cg = g.shape
    msE = H.shape[0]
    rowlen = H.shape[1]

    maxlen = 10
    econn = np.zeros((rg, maxlen + 1), dtype=int)
    econn[:, 0] = 1
    for k in range(msE):
        id = H[k, :]
        idlen = econn[id, 0]
        if np.max(idlen) == maxlen:
            maxlen += 10
            swap = np.zeros((rg, maxlen + 1), dtype=int)
            swap[:, 0:maxlen - 9] = econn
            econn = swap
        econn[id, 0] = idlen + 1
        for ii in range(rowlen):
            econn[id[ii], idlen[ii]] = k
    nodes = [ NODE(coord,[]) for coord in g]

    for k in range(rg):
        elen = econn[k, 0]
        nodes[k].ElementConnection = econn[k, 1:elen]

    return nodes

def MakeNode2dSmallFast(H, g):
    """
    Computes the Node data for MeshData.
    Node is a structure including all the nodal coordinates and
    for each node there is information to which nodes (NodeConnection) 
    and elements (ElementConnection) the node is
    connected.
    """

    rg, cg = g.shape
    msE = H.shape[0]
    rowlen = H.shape[1]

    maxlen = 10  # don't change
    econn = np.zeros((rg, maxlen + 1), dtype=np.uint32)
    econn[:, 0] = 1  # this will be incremented, each node is connected to at least two elements

    for k in range(msE):  # loop over elements
        id = H[k, :]  # node indices of this element
        idlen = econn[id, 0]  # current number of elements these nodes are connected to
        if np.max(idlen) == maxlen:  # more connected elements to one or more of these nodes than expected
            maxlen += 10
            swap = np.zeros((rg, maxlen + 1))
            swap[:, :maxlen - 9] = econn
            econn = swap
        econn[id, 0] = idlen + 1  # increment the connected elements count for these nodes
        for ii in range(rowlen):
            econn[id[ii], idlen[ii]] = k  # for these nodes, store the index of this connected element

    nodes = [ NODE(coord,[]) for coord in g]

    for k in range(rg):
        elen = econn[k, 0]
        nodes[k].ElementConnection = econn[k, 1:elen]

    return nodes

def FindElectrodeElements2_2D(H, Node, elnodes, order=1, gindx='yes'):
    """
    Returns:
        eltetra: list of lists containing indices to elements under each electrode
        E: ndarray with face indices if element i is under some electrode, zero otherwise
        nc: number of indices that were changed (related to gindx)
    """

    nc = 0

    # nJ = 2 (1st order elements), nJ = 3 (2nd order elements)
    nJ = 2  # default
    if order == 2:
        nJ = 3
    elif order != 1:
        print('order not supported, using default')

    nH = H.shape[0]
    Nel = len(elnodes)

    E = np.zeros((nH, nJ), dtype=np.uint32)
    eltetra = [None] * Nel

    tetra_mask = np.zeros(nH, dtype=np.uint8)

    # loop through every electrode
    for ii in range(Nel):
        ind_node = elnodes[ii]
        node_len = len(ind_node)

        # loop through nodes in electrode ii
        for jj in range(node_len):
            ind_tetra = Node[ind_node[jj]].ElementConnection
            tetra_len = len(ind_tetra)

            # check every tetrahedron connected to node ptr *ii
            for kk in range(tetra_len):
                tetra = ind_tetra[kk]
                Hind = H[tetra, :]

                if not tetra_mask[tetra]:  # ...we want to select the tetrahedron only once
                    C, II, JJ = np.intersect1d(Hind, ind_node, return_indices=True)
                    if len(C) == nJ:
                        eltetra[ii] = eltetra[ii] + [tetra] if eltetra[ii] else [tetra]

                        E[tetra, :] = np.sort(Hind[II])
                        if order == 2:
                            E[tetra, :] = E[tetra, [0, 2, 1]]
                    tetra_mask[tetra] = 1

    if gindx.lower() == 'yes' and order == 2:
        E, nc = reindex(E, Node)

    return eltetra, E, nc
#  Note that the returned data structure eltetra is now a list of lists instead of a cell array, which is more idiomatic in Python.

def reindex(E, Node):
    gN = len(Node)
    g = np.array([n.Coordinate for n in Node])  # Nodes
    nE = E.shape[0]

    nc = 0
    for ii in range(nE):
        if all(E[ii, :]):
            nodes = E[ii, :]
            mp = nodes[0:3]
            cp = 0.5 * (g[mp, :] + g[mp[np.array([1, 2, 0])], :])
            gg = g[nodes[3:6], :]  # center nodes (2nd order)
            I1 = np.argmin(np.sum((cp[0, :] - gg) ** 2, axis=1))
            I2 = np.argmin(np.sum((cp[1, :] - gg) ** 2, axis=1))
            I3 = np.argmin(np.sum((cp[2, :] - gg) ** 2, axis=1))
            nodes2 = np.hstack((mp, nodes[np.array([I1, I2, I3]) + 3]))
            E[ii, :] = nodes2
            if not np.array_equal(nodes, nodes2):
                nc += 1
    return E, nc

def MakeElement2dSmallCellFast(H, eltetra, E):
    rH, cH = H.shape
    nel = len(eltetra)

    Element = [ELEMENT(h,[])for h in H]

    for k in range(nel):
        ids = eltetra[k]
        for n in ids:
            Element[n].Electrode = [k, E[n, :]]

    return Element

def Reduce2ndOrderMesh_2D(H2, g2, elind2, format):
    Nel = len(elind2)

    if format == 1:
        H = H2[:, 0:3]
        ng = np.max(H)
        g = g2[0:ng, :]
        elind = [None] * Nel
        for ii in range(Nel):
            I = elind2[ii]
            J = np.where(I <= ng)[0]
            elind[ii] = I[J]
        Node = make_node_3d_small_fast(H, g)
        eltetra, E, nc = FindElectrodeElements2_2D(H, Node, elind, 1)
        Element = MakeElement2dSmallCellFast(H, eltetra, E)

    elif format == 2:
        H = H2[:, [0, 2, 4]]
        ng = np.max(H)
        g = g2[0:ng+1, :]
        elind = [None] * Nel
        for ii in range(Nel):
            I = elind2[ii]
            J = np.where(I <= ng)[0]
            elind[ii] = I[J]
        Node = make_node_3d_small_fast(H, g)
        eltetra, E, nc = FindElectrodeElements2_2D(H, Node, elind, 1)
        Element = MakeElement2dSmallCellFast(H, eltetra, E)

    return H, g, Node, Element, elind, eltetra, E

import numpy as np

def fixIndices2nd_2D(g, H, lns):
    Inds2nd = H[:, 3:].flatten()
    Inds2nd = np.unique(Inds2nd)

    Inds1st = H[:, 0:3].flatten()
    Inds1st = np.unique(Inds1st)

    gnew = np.vstack((g[Inds1st, :], g[Inds2nd, :]))

    Hnew = np.copy(H)
    for ii in range(H.shape[0]):
        for jj in range(3):  # main vertices
            ind = H[ii, jj]
            i = np.where(Inds1st == ind)[0]
            Hnew[ii, jj] = i
        for jj in range(3, 6):  # 2nd order vertices
            ind = H[ii, jj]
            i = np.where(Inds2nd == ind)[0]
            Hnew[ii, jj] = len(Inds1st) + i

    trisnew = np.copy(lns)
    for ii in range(lns.shape[0]):
        for jj in range(2):  # main vertices
            ind = lns[ii, jj]
            i = np.where(Inds1st == ind)[0]
            trisnew[ii, jj] = i
        for jj in range(2, 3):  # 2nd order vertices
            ind = lns[ii, jj]
            i = np.where(Inds2nd == ind)[0]
            trisnew[ii, jj] = len(Inds1st) + i

    return gnew, Hnew, trisnew


def create2Dmesh_circ(Nel, scaleparam, plotmesh=False, fignum=None):
    #This function creates a geo-file for Gmsh
    R = 1  # circle radius
    clscale = R / scaleparam

    filename = 'circ.geo'
    fname = 'circmesh'

    elwidth = 360 / (2 * Nel)

    elcenterangles = np.deg2rad(np.arange(elwidth / 2, 360, 2 * elwidth))
    elstartangles = elcenterangles - 0.5 * np.deg2rad(elwidth)
    gaplengths = np.diff(elstartangles) - np.deg2rad(elwidth)
    gaplengths = np.append(gaplengths, 2 * np.pi - (elstartangles[-1] + np.deg2rad(elwidth)) + elstartangles[0])

    elstartp = np.array([np.cos(elstartangles[0]) * R, np.sin(elstartangles[0]) * R])

    with open(filename, 'w') as fid:
        fid.write('SetFactory("OpenCASCADE");\n\n\n')

        fid.write(f'Point({1}) = {{{elstartp[0]}}},{elstartp[1]},0}};\n')
        for ii in range(1, 2 * Nel):
            if ii < 2 * Nel - 1:
                if ii % 2 == 1:  # creating an electrode
                    fid.write(f'Extrude {{0, 0, 1}}, {{0, 0, 0}}, -Pi/{{Nel}} {{Point{{{ii}}};}}\n')
                else:  # creating a gap between electrodes
                    fid.write(f'Extrude {{0, 0, 1}}, {{0, 0, 0}}, -{{gaplengths[ii // 2]}} {{Point{{{ii}}};}}\n')
            else:

                fid.write(f'Extrude {{0, 0, 1}}, {{0, 0, 0}}, -{{0.95 * gaplengths[-1]}} {{Point{{{ii}}};}}\n')
        # lastly, a line to connect the final point to the starting point
        fid.write(f'Line({2 * Nel + 1}) = {{{2 * Nel + 1},1}};\n')
        fid.write('Curve Loop(1) = {')
        for ii in range(1, 2 * Nel + 1):
            fid.write(f'{ii}, ')
        fid.write(f'{2 * Nel + 1}}};\n')
        fid.write('Plane Surface(1) = {1};\n')

        for ii in range(1, Nel + 1):
            fid.write(f'Physical Curve({ii}) = {{{(ii - 1) * 2 + 1}}};\n')
        fid.write(f'Physical Surface({50}) = {{{1}}};\n')
        fid.write('Mesh.SecondOrderLinear = 1;\n')  # important - no curved edges

    input('Create the mesh file using Gmsh, and then press any key to kontinue...')
    #os.system(f'gmsh {filename} -2 -order 2 -clscale {clscale} -format m -o {fname}.m')
    # Execute gmsh using the given command line. Make sure gmsh is installed and its path is correctly set.

    # Load the mesh file (.m) generated by gmsh here
    
    # ============================================================================
    # HERE YOU NEED A METHOD TO IMPORT THE MESH FILE PRODUCED BY GMSH TO PYTHON. AFTER
    # YOU HAVE IT IN THE VARIABLE msh, YOU CAN CONTINUE
    # ============================================================================

    g2 = msh.POS[:, :2]
    H2 = msh.TRIANGLES6[:, :6]
    lns2 = msh.LINES3[:, :3]

    # Replace msh.POS, msh.TRIANGLES6, and msh.LINES3 with the corresponding variables obtained from the gmsh-generated mesh

    rmat = np.array([[np.cos(np.deg2rad(90)), -np.sin(np.deg2rad(90))], [np.sin(np.deg2rad(90)), np.cos(np.deg2rad(90))]])
    g2 = np.matmul(g2, rmat.T)
    g2[:,0] = -g2[:,0]

    g2, H2, lns2 = fixIndices2nd_2D(g2, H2, lns2)
    H2 = H2[:, [0, 3, 1, 4, 2, 5]]
    lns2 = lns2[:, [0, 2, 1]]

    elfaces = []
    elfaces2 = []
    elind2 = []
    for ii in range(Nel):
        tris = np.where(msh.LINES3[:, -1] == ii)[0]
        elfaces2.append(lns2[tris, :3])
        elind2.append(np.unique(elfaces2[-1].flatten()))
        elfaces.append(lns2[tris, [[0], [2]]])

    Node2 = MakeNode2dSmallFast(H2, g2)
    eltetra2, E2, nc = FindElectrodeElements2_2D(H2, Node2, elind2, 2, "No")
    Element2 = MakeElement2dSmallCellFast(H2, eltetra2, E2)
    H, g, Node, Element, elind, eltetra, E = Reduce2ndOrderMesh_2D(H2, g2, elind2, 2)

    if plotmesh:
        plt.figure(fignum)
        plt.clf()
        plt.triplot(g[:, 0], g[:, 1], H[:, :3])
        plt.axis('image')
        plt.ion()
        plt.show()
        
        #plt.hold(True)

        for ii in range(Nel):
            inds = elfaces2[ii].flatten()
            nds = g2[inds, :]
            if ii % 2 == 0:
                plt.plot(nds[:, 0], nds[:, 1], 'o', color='r', markerfacecolor='r')
            else:
                plt.plot(nds[:, 0], nds[:, 1], 'o', color='m', markerfacecolor='m')

        plt.pause(0.001)
        #plt.gca().set_position([0.3, 0.6, 0.3, 0.4])
        
    mesh = Mesh(H,g,elfaces,Node,Element)
    mesh2 = Mesh(H2,g2,elfaces2,Node2,Element2)

    #mesh2 = {'H': H2, 'g': g2, 'elfaces': elfaces2, 'Node': Node2, 'Element': Element2}
    #mesh = {'H': H, 'g': g, 'elfaces': elfaces, 'Node': Node, 'Element': Element}

    return mesh2, mesh, elcenterangles