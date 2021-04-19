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


class AlphaComplex:

    def __init__(self, X):
        '''
        :param X:
        '''
        self.filtration_dict = {}
        self.filtration_backup = {}
        self.dimension = X.shape[1]
        self.generators = {}
        # self.pairs = []

        DT = Delaunay(X)
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
                    P = X[simplex, :]
                    if math.isnan(node.filtration):
                        _, node.filtration = circumsphere(P)

                    # computing facets filtration values
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
                            Q = P[:, :]
                            q = Q[cnt, :]
                            Q = np.delete(Q, cnt, axis=0)
                            if subsimplex_key not in self.filtration_dict:
                                self.filtration_dict[subsimplex_key] = circumsphere(Q)
                            c, r = self.filtration_dict[subsimplex_key]
                            dist = np.sum(np.power(q - c, 2))
                            if dist < r: # if not Gabriel
                                face.filtration = node.filtration
                        cnt += 1


    def persistent_homology_in_dimension(self, dim, canonical=0):
        '''
        :param dim: homology dimension to be calculated
        :return: persistent homology intervals, generators and simplexes pairings
        '''
        intervals, generators, neg2pos_dict, pos2idx = self.ST.persistent_homology_in_dimension(dim, canonical)
        return intervals, generators, neg2pos_dict, pos2idx
