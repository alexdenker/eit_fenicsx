"""
Written by KTC2023 challenge organizers:
https://zenodo.org/record/8252370

"""

import numpy as np
import scipy as sp

from .KTCMeshing import ELEMENT

class CMATRIX:# Compact presentation of a very sparse matrix
    def __init__(self, mat, indi, indj=None):
        self.mat = mat
        self.i = indi
        if indj != None:
            self.j = indj
        else:
            self.j = indi
        

class EITFEM:
    def __init__(self, Mesh2, Inj, Mpat=None, vincl=None, sigmamin=None, sigmamax=None):
        self.Mesh2 = Mesh2
        self.Inj = Inj
        self.Mpat = Mpat
        self.sigmamin = sigmamin if sigmamin is not None else 1e-9
        self.sigmamax = sigmamax if sigmamax is not None else 1e9
        self.zmin = 1e-6
        self.Nel = len(Mesh2.elfaces)
        self.nH2 = Mesh2.H.shape[0]
        self.ng2 = Mesh2.g.shape[0]
        
        self.vincl = vincl if vincl is not None else np.ones((len(Mesh2)*len(Mesh2)), dtype=bool)
        self.mincl = np.array(vincl)
        self.mincl.shape = (self.Nel-1,self.Inj.shape[1])
        

        self.dA = None
        self.C = np.matrix(np.vstack((np.ones((1, self.Nel-1)), -np.eye(self.Nel-1))))

    def SolveForward(self, sigma, z):
        # Forward solution function - solves the potential field and
        # measured voltages given conductivity sigma and contact impedances
        # z

        # Project negative / too small sigma values to a given minimum
        # Project too high sigma values to a given maximum
        sigma[sigma < self.sigmamin] = self.sigmamin
        sigma[sigma > self.sigmamax] = self.sigmamax
        z[z < self.zmin] = self.zmin

        # Compute the conductivity-dependent part of the FEM matrix
        gN = max(np.shape(self.Mesh2.Node))
        HN = max(np.shape(self.Mesh2.H))  # number of elements
        k = 1
        Arow = np.zeros((6 * HN, 6), dtype=np.uint32)
        Acol = np.zeros((6 * HN, 6), dtype=np.uint32)
        Aval = np.zeros((6 * HN, 6))
        for ii in range(HN):  # Go through all triangles
            ind = self.Mesh2.H[ii, :]
            gg = self.Mesh2.g[ind, :]
            ss = sigma[ind[[0, 2, 4]]]
            int = self.grinprod_gauss_quad_node(gg, ss)
            
            Inds1 = np.tile(ind, (6, 1))
            Acol[k-1:k+5, :] = Inds1
            ind = ind.T


            Inds2 = Inds1.T
            Arow[k-1:k+5, :] = Inds2
            Aval[k-1:k+5, :] = int
            k = k + 6
        A0 = sp.sparse.csr_matrix((Aval.flatten(), (Arow.flatten(), Acol.flatten())), shape=(self.ng2+self.Nel-1, self.ng2+self.Nel-1))

        # Compute the rest of the FEM matrix
        M = sp.sparse.csr_matrix((gN, self.Nel))
        K = sp.sparse.csr_matrix((gN, gN))
        s = np.zeros((self.Nel, 1))
        g = self.Mesh2.g  # Nodes
        for ii in range(HN):
            # Go through all triangles
            ind = self.Mesh2.Element[ii].Topology  # The indices to g of the ii'th triangle.
            if self.Mesh2.Element[ii].Electrode:  # Checks if the triangle ii is under an electrode
                Ind = self.Mesh2.Element[ii].Electrode[1]
                a = g[Ind[0], :]
                b = g[Ind[1], :]  # the 2nd order node
                c = g[Ind[2], :]
                InE = self.Mesh2.Element[ii].Electrode[0]  # Electrode index.
                s[InE] = s[InE] + 1 / z[InE] * self.electrlen(np.array([a, c]))  # Assumes straight electrodes.
                bb1 = self.bound_quad1(np.array([a, b, c]))
                bb2 = self.bound_quad2(np.array([a, b, c]))
                for il in range(6):
                    eind = np.where(self.Mesh2.Element[ii].Topology[il] == self.Mesh2.Element[ii].Electrode[1])[0]
                    if eind.size != 0:
                        M[ind[il], InE] = M[ind[il], InE] - 1 / z[InE] * bb1[eind[0]]
                    for im in range(6):
                        eind1 = np.where(self.Mesh2.Element[ii].Topology[il] == self.Mesh2.Element[ii].Electrode[1])[0]
                        eind2 = np.where(self.Mesh2.Element[ii].Topology[im] == self.Mesh2.Element[ii].Electrode[1])[0]
                        if eind1.size != 0 and eind2.size != 0:
                            K[ind[il], ind[im]] = K[ind[il], ind[im]] + 1 / z[InE] * bb2[eind1[0], eind2[0]]

        tS = sp.sparse.csr_matrix(np.diag(s.flatten()))
        S = sp.sparse.csr_matrix(self.C.T * tS * self.C)
        M = M * self.C
        #S0 = sp.sparse.csr_matrix(np.block([[K.toarray(), M.toarray()], [M.toarray().T, S.toarray()]]))
        S0 = sp.sparse.bmat(
            [
            [K, M],
            [M.T, S]
            ])
            

        self.A = A0 + S0
        self.b = np.concatenate((np.zeros((self.ng2, self.Inj.shape[1])), self.C.T * self.Inj), axis=0)  # RHS vector
        UU = sp.sparse.linalg.spsolve(self.A, self.b)  # Solve the FEM system of equations
        self.theta = UU  # FEM solution vectors for each current injection
        self.Pot = UU[0:self.ng2, :]  # The electric potential field for each current injection
        self.Imeas = self.Inj  # Injected currents
        self.Umeas = self.Mpat.T * self.C * self.theta[self.ng2:, :]  # Measured voltages between electrodes
        self.Umeas = self.Umeas.T[self.mincl.T].T
        self.QC = np.block([[np.zeros((np.shape(self.Mpat)[1], self.ng2)), self.Mpat.T @ self.C]])
        fsol = self.Umeas  # Measured potentials, vectorized

        return fsol

    def Jacobian(self, sigma=None, z=None):
        if sigma is not None and z is not None:
            self.SolveForward(sigma, z)
        else:
            if self.theta is None:
                raise ValueError("Jacobian cannot be computed without first computing a forward solution")

        if self.dA is None:
            temp_elements = []
            for element in self.Mesh2.Element:
                temp_element = ELEMENT(element.Topology[[0, 2, 4]], element.Electrode)
                temp_elements.append(temp_element)
            self.dA = self.jacob_nodesigma2nd3(self.Mesh2.Node, temp_elements, self.Mesh2.Node, self.Mesh2.Element)

        m = len(sigma)
        n = len(self.Umeas)

        Jleft = sp.sparse.linalg.spsolve(self.A.T, self.QC.T)
        Jright = self.theta

        Js = np.zeros((n, m))
        for ii in range(m):
            Jid = self.dA[ii].i # non-zero elements of dA
            Jtemp = -Jleft.T[:, Jid] @ self.dA[ii].mat @ Jright[Jid, :]
            Js[:, ii] = Jtemp.T[self.mincl.T] # remove elements where are no measurments

        return Js

    def Jacobianz(self, sigma=None, z=None):
        if sigma is not None and z is not None:
            self.SolveForward(sigma, z)
        else:
            if self.theta is None:
                raise ValueError("Jacobian cannot be computed without first computing a forward solution")

        dA_dz = self.ComputedA_dz( z)

        m = len(z)
        n = len(self.Umeas)

        #Jleft = -self.QC / self.A
        Jleft = -self.QC @ sp.sparse.linalg.inv(self.A)
        Jright = self.theta

        Jz = np.zeros((n, m))
        for ii in range(m):
            Jtemp = Jleft @ dA_dz[ii] @ Jright
            Jz[:, ii] = Jtemp.T[self.mincl.T]

        return Jz

    def SetInvGamma(self, noise_percentage, noise_percentage2=0, meas_data=None):
        if meas_data is not None:
            meas = meas_data
        else:
            meas = self.Umeas

        var_meas = np.power(((noise_percentage / 100) * (np.abs(meas))),2)
        var_meas = var_meas + np.power((noise_percentage2 / 100) * np.max(np.abs(meas)),2)
        Gamma_n = np.diag(np.array(var_meas).flatten())

        InvGamma_n = np.linalg.inv(Gamma_n)
        self.InvGamma_n = sp.sparse.csr_matrix(InvGamma_n)
        self.Ln = sp.sparse.csr_matrix(np.linalg.cholesky(InvGamma_n))
        self.InvLn = sp.sparse.csr_matrix(np.linalg.cholesky(Gamma_n))
        q = 0

    def grinprod_gauss_quad_node(self, g, sigma):
        w = np.array([1/6, 1/6, 1/6])
        ip = np.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])

        int_sum = 0
        for ii in range(3):
            S = np.array([1 - ip[ii, 0] - ip[ii, 1], ip[ii, 0], ip[ii, 1]])
            L = np.array([[4 * (ip[ii, 0] + ip[ii, 1]) - 3, -8 * ip[ii, 0] - 4 * ip[ii, 1] + 4, 4 * ip[ii, 0] - 1, 4 * ip[ii, 1], 0, -4 * ip[ii, 1]],
                          [4 * (ip[ii, 0] + ip[ii, 1]) - 3, -4 * ip[ii, 0], 0, 4 * ip[ii, 0], 4 * ip[ii, 1] - 1, -8 * ip[ii, 1] - 4 * ip[ii, 0] + 4]])
            Jt = L @ g
            iJt = np.linalg.inv(Jt)
            dJt = np.abs(np.linalg.det(Jt))
            G = iJt @ L
            int_sum += w[ii] * (S.T @ sigma) * G.T @ G * dJt

        return int_sum

    def bound_quad1(self, g):
        w = np.array([1/2, 1/2])
        ip = np.array([1/2 - 1/6 * np.sqrt(3), 1/2 + 1/6 * np.sqrt(3)])

        int_sum = 0
        for ii in range(2):
            S = np.array([2 * ip[ii]**2 - 3 * ip[ii] + 1,
                          -4 * ip[ii]**2 + 4 * ip[ii],
                          2 * ip[ii]**2 - ip[ii]])
            dJt = np.sqrt((g[0, 0] * (4 * ip[ii] - 3) + g[1, 0] * (4 - 8 * ip[ii]) + g[2, 0] * (4 * ip[ii] - 1))**2 +
                          (g[0, 1] * (4 * ip[ii] - 3) + g[1, 1] * (4 - 8 * ip[ii]) + g[2, 1] * (4 * ip[ii] - 1))**2)
            int_sum += w[ii] * S * dJt

        return int_sum

    def bound_quad2(self, g):
        w = np.array([5/18, 8/18, 5/18])
        ip = np.array([1/2 - 1/10 * np.sqrt(15), 1/2, 1/2 + 1/10 * np.sqrt(15)])

        int_sum = np.zeros((3,3))
        for ii in range(3):
            S = np.array([2 * ip[ii]**2 - 3 * ip[ii] + 1,
                          -4 * ip[ii]**2 + 4 * ip[ii],
                          2 * ip[ii]**2 - ip[ii]])
            dJt = np.sqrt((g[0, 0] * (4 * ip[ii] - 3) + g[1, 0] * (4 - 8 * ip[ii]) + g[2, 0] * (4 * ip[ii] - 1))**2 +
                          (g[0, 1] * (4 * ip[ii] - 3) + g[1, 1] * (4 - 8 * ip[ii]) + g[2, 1] * (4 * ip[ii] - 1))**2)
            temp = np.outer(S,S)
            int_sum += w[ii] * np.outer(S,S) * dJt

        return int_sum

    def electrlen(self, g):
        dJt = np.sqrt((g[1, 0] - g[0, 0])**2 + (g[1, 1] - g[0, 1])**2)
        return dJt

    def jacob_nodesigma2nd3(self, Nodesig, Elementsig, Node, Element):# Returns a CMATRIX list representing the derivative of the stiffness matrix w.r.t. coefficients of to the conductivity 
        gNs = len(Nodesig)
        gN = len(Node)

        Agrads = []

        for jj in range(gNs):
            nzi = np.array([],dtype= np.uint32)
            nzj = np.array([],dtype= np.uint32)
            nzv = np.array([],dtype= np.float64)
            indexmap = np.zeros((gN,1),dtype= np.uint32)
            Aa = sp.sparse.csr_matrix((gN, gN))
            El = Nodesig[jj].ElementConnection


            for ii in range(len(El)):
                indsig = Elementsig[El[ii]].Topology
                ind = Element[El[ii]].Topology
                gg = np.array([Node[i].Coordinate for i in ind]).reshape(6, 2)
                I = np.where(jj == np.array(indsig))


                if len(I[0]) > 0:
                    anis = self.grinprodgaus2ndnode3(gg, I[0][0])

                    tmpind = np.tile(ind, (6,1))
                    nzi = np.append(nzi,tmpind.T.flatten())
                    nzj = np.append(nzj,tmpind.flatten())
                    nzv = np.append(nzv,anis.flatten())

            unzi = np.unique(nzi)
            nunz = len(unzi)
            tmat = np.zeros((nunz,nunz))
            t = np.array(range(nunz))
            t.shape = (nunz,1)
            indexmap[unzi] = t
            for ii in range(len(nzi)):
                ti = indexmap[nzi[ii]]
                tj = indexmap[nzj[ii]]
                tmat[ti,tj] += nzv[ii]
            Agrads.append(CMATRIX(tmat,unzi))
            #Agrad[:, jj] = np.array(Aa).flatten()

        J = Agrads
        return J

    def grinprodgaus2ndnode3(self, g, I):
        w = np.array([1/40] * 3 + [1/15] * 3 + [27/120])
        ip = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1/2, 0],
                       [1/2, 1/2],
                       [0, 1/2],
                       [1/3, 1/3]])

        int_sum = 0
        for ii in range(7):
            S = np.array([1 - ip[ii, 0] - ip[ii, 1], ip[ii, 0], ip[ii, 1]])
            L = np.array([[4 * (ip[ii, 0] + ip[ii, 1]) - 3, -8 * ip[ii, 0] - 4 * ip[ii, 1] + 4,
                           4 * ip[ii, 0] - 1, 4 * ip[ii, 1], 0, -4 * ip[ii, 1]],
                          [4 * (ip[ii, 0] + ip[ii, 1]) - 3, -4 * ip[ii, 0],
                           0, 4 * ip[ii, 0], 4 * ip[ii, 1] - 1, -8 * ip[ii, 1] - 4 * ip[ii, 0] + 4]])
            Jt = L @ g
            iJt = np.linalg.inv(Jt)
            dJt = abs(np.linalg.det(Jt))
            G = iJt @ L
            int_sum += w[ii] * S[I] * G.T @ G * dJt

        return int_sum

    def regroupAgrad2cell(self, Agrad, ng=None, ngs=None):
        if ng is None and ngs is None:
            ng = int(np.sqrt(Agrad.shape[0]))
            ngs = Agrad.shape[1]

        arow = [None] * ng
        acol = [None] * ng
        aval = [None] * ng

        for k in range(ngs):
            S = np.array(Agrad[:, k]).reshape(ng, ng)
            I, J = np.nonzero(S)

            I = np.unique(I)
            J = np.unique(J)

            arow[k] = I
            acol[k] = J
            aval[k] = S[np.ix_(I, J)]

        return arow, acol, aval

    def ComputedA_dz(self, z):
        dA_dz = [None] * self.Nel
        gN = len(self.Mesh2.Node)
        HN = len(self.Mesh2.H)

        M = sp.sparse.lil_matrix((gN, self.Nel))
        K = [sp.sparse.lil_matrix((gN, gN)) for _ in range(self.Nel)]
        s = np.zeros(self.Nel)
        g = np.array([node.Coordinate for node in self.Mesh2.Node]).reshape(gN, 2)

        for ii in range(HN):
            ind = self.Mesh2.Element[ii].Topology

            if len(self.Mesh2.Element[ii].Electrode) > 0:
                Ind = self.Mesh2.Element[ii].Electrode[1]
                a = g[Ind[0]]
                b = g[Ind[1]]
                c = g[Ind[2]]
                InE = self.Mesh2.Element[ii].Electrode[0]
                s[InE] -= (1 / (z[InE]**2)) * self.electrlen(np.array([a, c]))

                bb1 = self.bound_quad1(np.array([a, b, c]))
                bb2 = self.bound_quad2(np.array([a, b, c]))

                for il in range(6):
                    eind = np.where(self.Mesh2.Element[ii].Topology[il] == self.Mesh2.Element[ii].Electrode[1])[0]

                    if eind.size > 0:
                        M[ind[il], InE] += (1 / (z[InE]**2)) * bb1[eind[0]]

                    for im in range(6):
                        eind1 = np.where(self.Mesh2.Element[ii].Topology[il] == self.Mesh2.Element[ii].Electrode[1])[0]
                        eind2 = np.where(self.Mesh2.Element[ii].Topology[im] == self.Mesh2.Element[ii].Electrode[1])[0]

                        if eind1.size > 0 and eind2.size > 0:
                            K[InE][ind[il], ind[im]] -= (1 / (z[InE]**2)) * bb2[eind1[0], eind2[0]]

        for ii in range(self.Nel):
            tmp = np.zeros(self.Nel)
            tmp[ii] = s[ii]
            S = sp.sparse.diags(tmp)
            S = self.C.T @ S @ self.C
            #Mtmp = np.zeros_like(M)
            Mtmp = sp.sparse.lil_matrix((gN, self.Nel))
            Mtmp[:, ii] = M[:, ii]
            Mtmp = Mtmp @ self.C
            dA_dz[ii] = np.block([[K[ii].toarray(), Mtmp],
                                  [Mtmp.T, S]])

        return dA_dz





