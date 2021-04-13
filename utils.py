import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np


# def cycle_prevalence():


def list2str(list):
    str_list = [str(x) for x in list]
    return '-'.join(str_list)


def str2list(str):
    x = str.split('-')
    list = [int(a) for a in x]
    return list


def gentorus(params):
    r1, r2, N = params
    X = np.zeros((N, 3))
    for i in range(N):
        phi = 2*np.pi*np.random.rand(1)
        theta = 2*np.pi*np.random.rand(1)
        w = np.random.rand(1)
        th = r1 + r2 * np.cos(theta)/(2*np.pi*r1)
        while w > th:
            phi = 2 * np.pi * np.random.rand(1)
            theta = 2 * np.pi * np.random.rand(1)
            w = np.random.rand(1)
        x = (r1 + r2 * np.cos(theta)) * np.cos(phi)
        y = (r1 + r2 * np.cos(theta)) * np.sin(phi)
        z = r2 * np.sin(theta)
        X[i, :] = [x, y, z]
    return X


def plot_complex3D(X, simplex_tree, r=float('inf')):

    simplex_dict = simplex_tree.simplex_dict
    vertices = [node.val for node in simplex_dict[0]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[vertices, 0], X[vertices, 1], X[vertices, 2], s=5)

    edges = [simplex_tree.node2simplex(node) for node in simplex_dict[1] if node.filtration <= r]
    triangles = [simplex_tree.node2simplex(node) for node in simplex_dict[2] if node.filtration <= r]

    linewidth = 0.5
    for e in edges:
        ax.plot(X[e, 0], X[e, 1], X[e, 2], 'k', linewidth=linewidth)

    for t in triangles:
        points = X[t, :]
        poly3d = Poly3DCollection(points, linewidths=linewidth, facecolors='y', alpha=0.05)
        poly3d.set_edgecolor('k')
        ax.add_collection3d(poly3d)


def plot_complex(X, simplex_tree, r=float('inf')):

    simplex_dict = simplex_tree.simplex_dict
    vertices = [node.val for node in simplex_dict[0]]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X[vertices, 0], X[vertices, 1], s=5)

    edges = [simplex_tree.node2simplex(node) for node in simplex_dict[1] if node.filtration <= r]
    triangles = [simplex_tree.node2simplex(node) for node in simplex_dict[2] if node.filtration <= r]

    linewidth = 0.5
    for e in edges:
        ax.plot(X[e, 0], X[e, 1], 'k', linewidth=linewidth)

    for t in triangles:
        points = X[t, :]
        path = Path(points)
        patch = PathPatch(path, facecolor='y', edgecolor='k', linewidth=linewidth, alpha=0.1)
        ax.add_patch(patch)


def plot_1D_generators(X, g, d):
    if d == 2:
        for v in g:
            x = X[v, 0]
            y = X[v, 1]
            plt.plot(x, y, 'r', linewidth=3)
    if d == 3:
        for v in g:
            x = X[v, 0]
            y = X[v, 1]
            z = X[v, 2]
            plt.plot(x, y, z, 'r', linewidth=3)
