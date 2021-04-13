import numpy as np
import scipy as sp
from simplex_tree import SimplexTree
from scipy.spatial import Delaunay
import math
from utils import list2str


def circumsphere(P):
    '''
    :param P: array Nxd of N points of dimension d
    :return: center and squared radius of the circum-sphere of P
    '''
    p1 = P[0, :]
    A = P[1:, :] - p1
    Pnorm = np.sum(np.power(P, 2), 1)
    b = 0.5*(Pnorm[1:] - Pnorm[0])
    invA = np.linalg.pinv(A)
    c0 = invA.dot(b)
    F = sp.linalg.null_space(A)
    if F.size != 0:
        z = np.transpose(F).dot(p1-c0)
        c = c0 + F.dot(z)
    else:
        c = c0
    R = np.sum(np.power(p1-c, 2))
    return c, R


def filtration_value(Q, P):
    if Q.size == 0:
        c, R = circumsphere(P)
        Rx = R
        Ry = R
    elif P.size == 0:
        c, R = circumsphere(Q)
        Rx = R
        Ry = R
    elif Q.shape[0] == 1 and P.shape[0] == 1:
        c = 0.5*(Q + P)
        R = np.sum(np.power(0.5*(Q - P), 2))
        Rx = R
        Ry = R
    else:
        q1 = Q[0, :]
        p1 = P[0, :]
        Qdiff = Q[1:, :] - Q[0, :]
        Pdiff = P[1:, :] - P[0, :]
        A = np.concatenate((Qdiff, Pdiff), 0)
        Qnorm = np.sum(np.power(Q, 2), 1)
        Pnorm = np.sum(np.power(P, 2), 1)
        b = 0.5*np.concatenate((Qnorm[1:]-Qnorm[0], Pnorm[1:]-Pnorm[0]), 0)
        x0 = np.linalg.pinv(A).dot(b)
        F = sp.linalg.null_space(A)
        if F.size != 0:
            zq = np.transpose(F).dot(q1 - x0)
            zp = np.transpose(F).dot(p1 - x0)
            xq = x0 + F.dot(zq)
            xp = x0 + F.dot(zp)
            rqq = np.sum(np.power(q1-xq, 2))
            rqp = np.sum(np.power(p1-xq, 2))
            rpq = np.sum(np.power(q1-xp, 2))
            rpp = np.sum(np.power(p1-xp, 2))
        else:
            xq = x0
            xp = x0
            rqq = np.sum(np.power(q1-x0, 2))
            rqp = np.sum(np.power(p1-x0, 2))
            rpq = rqq
            rpp = rqp
        if rqq >= rqp:
            c = xq
            Rx = rqq
            Ry = rqp
            R = Rx
        elif rpp >= rpq:
            c = xp
            Rx = rpq
            Ry = rpp
            R = Ry
        else:
            c, R = circumsphere(np.concatenate((Q, P), 0))
            Rx = R
            Ry = R

    return c, R, Rx, Ry


class CoupledAlpha:

    def __init__(self, X, Y, l=1):
        '''
        :param X: numpy array
        :param Y: numpy array
        :param l: distance between hyperplanes
        '''
        self.filtration_dict = {}
        self.filtration_backup = {}
        self.dimension = X.shape[1]
        self.generators = {}
        self.len_X = X.shape[0]
        self.len_Y = Y.shape[0]
        self.Xsimplexesdict = {}
        self.Ysimplexesdict = {}
        self.Zsimplexesdict = {}
        for i in range(self.dimension+2):
            self.Xsimplexesdict[i] = []
            self.Ysimplexesdict[i] = []
            self.Zsimplexesdict[i] = []
        # sets lifting
        XX = np.concatenate((X, np.zeros((self.len_X, 1))), 1)
        YY = np.concatenate((Y, l*np.ones((self.len_Y, 1))), 1)
        Z = np.concatenate((XX, YY), 0)
        # Delaunay triangulation
        DT = Delaunay(Z)
        S = DT.simplices.tolist()
        self.ST = SimplexTree(S)

        # setting the filtration values of all simplexes to null
        for k in range(1, len(self.ST.lists)):
            k_lists = self.ST.lists[k]
            for key in k_lists:
                for node in k_lists[key]:
                    node.filtration = float('nan')

        # computing the filtration values in a top down fashion
        for k in range(len(self.ST.lists)-1, 0, -1):
            k_lists = self.ST.lists[k]
            for key in k_lists:
                for node in k_lists[key]:
                    simplex = self.ST.node2simplex(node)

                    # computing simplex filtration value
                    Px_idx = [i for i in simplex if i < self.len_X]
                    Py_idx = [i-self.len_X for i in simplex if i >= self.len_X]
                    Px = X[Px_idx, :]
                    Py = Y[Py_idx, :]
                    if math.isnan(node.filtration):
                        _, node.filtration, _, _ = filtration_value(Px, Py)

                    # constructing dictionaries for fast computation of image homology
                    if simplex[-1] < self.len_X:
                        self.Xsimplexesdict[k].append([node, node.filtration])
                    elif simplex[0] >= self.len_X:
                        self.Ysimplexesdict[k].append([node, node.filtration])
                    else:
                        self.Zsimplexesdict[k].append([node, node.filtration])

                    # computing simplex facets filtration values
                    if k == 1:
                        continue
                    facets = self.ST.get_facets(simplex)
                    cnt = 0
                    for face in facets:
                        if not math.isnan(face.filtration):
                            face.filtration = min(node.filtration, face.filtration)
                        else: # check if coupled Gabriel
                            idx = simplex[cnt]  # the index of the point not included in the facet
                            subsimplex = [i for i in simplex if i != idx]
                            subsimplex.sort()
                            subsimplex_key = list2str(subsimplex)
                            if idx < self.len_X:
                                Qx = Px[:, :]
                                q = Qx[cnt, :]
                                Qx = np.delete(Qx, cnt, axis=0)
                                Qy = Py[:, :]
                            else:
                                Qx = Px[:, :]
                                Qy = Py[:, :]
                                q = Qy[cnt - Px.shape[0], :]
                                Qy = np.delete(Qy, cnt - Px.shape[0], axis=0)
                            if subsimplex_key not in self.filtration_dict:
                                self.filtration_dict[subsimplex_key] = filtration_value(Qx, Qy)
                            c, R, Rx, Ry = self.filtration_dict[subsimplex_key]
                            dist = np.sum(np.power(q - c, 2))
                            if idx < self.len_X:
                                r = Rx
                            else:
                                r = Ry
                            if dist < r: # if not coupled Gabriel
                                face.filtration = node.filtration
                        cnt += 1


    def persistent_homology_in_dimension(self, dim, image_flag=-1):
        '''
        :param dim: dimension of persistent homology
        :param image_flag: if 0 computes the image homology for X if 1 for Y
        :return: intervals - pairs of (birth, death), generators - list of generators, neg2pos_dict - negative simplex
        to positive simplex dictionary, pos2idx - positive simplex to index in intervals (generators) list
        '''
        if image_flag == 0: # image homology of X
            for s in self.Ysimplexesdict[dim]:
                s[0].filtration = float('inf')
            for s in self.Zsimplexesdict[dim]:
                s[0].filtration = float('inf')
        elif image_flag == 1:
            for s in self.Xsimplexesdict[dim]:
                s[0].filtration = float('inf')
            for s in self.Zsimplexesdict[dim]:
                s[0].filtration = float('inf')

        # computing image homology
        intervals, generators, neg2pos_dict, pos2idx = self.ST.persistent_homology_in_dimension(dim)

        # reversing the infinite filtration values to the original values
        if image_flag == 0: # image homology of X
            for s in self.Ysimplexesdict[dim]:
                s[0].filtration = s[1]
            for s in self.Zsimplexesdict[dim]:
                s[0].filtration = s[1]
        elif image_flag == 1: # image homology of Y
            for s in self.Xsimplexesdict[dim]:
                s[0].filtration = s[1]
            for s in self.Zsimplexesdict[dim]:
                s[0].filtration = s[1]

        return intervals, generators, neg2pos_dict, pos2idx
