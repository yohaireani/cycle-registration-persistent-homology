import numpy as np
import matplotlib.pyplot as plt
from coupled_alpha import CoupledAlpha
from alpha_complex import AlphaComplex
from utils import *
import math
import concurrent.futures
from math import log
import time


def loop(params):
    i, X1, hom_dim, pos2idx_X1, intervals, gen_func, gen_params = params

    print('iteration number: ' + str(i + 1))
    X2 = gen_func(gen_params) # new sample
    alpha_X2 = AlphaComplex(X2)
    intervals_X2, _, _, pos2idx_X2 = alpha_X2.persistent_homology_in_dimension(hom_dim)

    # Computing the coupled alpha complex and the image homology for both X1 and X2
    co_alpha = CoupledAlpha(X1, X2)
    intervals_image_X1, _, neg2pos_image_X1, pos2idx_image_1 = co_alpha.persistent_homology_in_dimension(hom_dim,
                                                                                                        image_flag=0)

    intervals_image_X2, _, neg2pos_image_X2, pos2idx_image_2 = co_alpha.persistent_homology_in_dimension(hom_dim,
                                                                                                        image_flag=1)

    # finding matching cycles between X1 and X2
    score = np.zeros((1, len(pos2idx_X1)))
    for key in neg2pos_image_X1:
        idx_1 = pos2idx_X1[neg2pos_image_X1[key]]
        b_1 = intervals[idx_1][0]
        d_1 = intervals[idx_1][1]
        if key in neg2pos_image_X2:
            idx_image_1 = pos2idx_image_1[neg2pos_image_X1[key]]
            idx_2 = pos2idx_X2[neg2pos_image_X2[key]]
            idx_image_2 = pos2idx_image_2[neg2pos_image_X2[key]]
            b_2 = intervals_X2[idx_2][0]
            d_2 = intervals_X2[idx_2][1]
            b_im_1 = intervals_image_X1[idx_image_1][0]
            d_im_1 = intervals_image_X1[idx_image_1][1]
            b_im_2 = intervals_image_X2[idx_image_2][0]
            d_im_2 = intervals_image_X2[idx_image_2][1]
            # cycle prevalence score
            a1 = (min(d_1, d_im_1) - max(b_1, b_im_1)) / (max(d_1, d_im_1) - min(b_1, b_im_1))
            a2 = (min(d_2, d_im_2) - max(b_2, b_im_2)) / (max(d_2, d_im_2) - min(b_2, b_im_2))
            c12 = (min(d_1, d_2) - max(b_1, b_2)) / (max(d_1, d_2) - min(b_1, b_2))
            score[0, idx_1] = a1*a2*c12

    return score


# point cloud generator function
def gen_func(gen_params):
    modelFlag = gen_params[-1]
    if modelFlag == 1:
        return gentorus(gen_params[0:-1])
    elif modelFlag == 2:
        return genclusters(gen_params[0:-1])


def resample_func(gen_params):
    [X1, bw] = gen_params
    return resample_gauss(X1, bw)


if __name__ == '__main__':

    start = time.time()

    modelFlag = 1 # 1 - torus; 2 - clusters
    resampleFlag = 0 # 0 - resample from original distribution; 1 - use kernel density estimator
    # number of points in a sample
    N = 100
    # number of resamples
    B = 100
    # dimension of homology
    hom_dim = 1
    # gen_func parameters
    if modelFlag == 1: # torus
        d = 3 # dimension
        r1 = 0.5 # radius from origin to center of tube
        r2 = 0.2 # radius of tube
        gen_params = [r1, r2, N, modelFlag]
        # resample parameters
        bw = 0.001
        # filtration value for figures
        r = 0.25
        r = r ** 2
        # persistence diagram figure upper limit
        M = 0.3*r1
    elif modelFlag == 2: # clusters
        d = 2 # dimension of space
        numClusters = 5 # number of clusters
        cycleRadius = 0.1 # radius of the cycle supported on clusters
        clusterRadius = 0.01 # radius of each cluster
        p = 1 # percentage of noise
        gen_params = [N, numClusters, cycleRadius, clusterRadius, p, modelFlag]
        # resample parameters
        bw = 0.00002
        # filtration value for figures
        r = 0.0
        r = r ** 2
        # persistence diagram figure upper limit
        M = 2*cycleRadius

    X1 = gen_func(gen_params)
    dim = X1.shape[1]
    alpha = AlphaComplex(X1)
    intervals, generators, _, pos2idx_X1 = alpha.persistent_homology_in_dimension(hom_dim)
    # resample_func parameters
    resample_params = [X1, bw]
    # list of params for the multiprocessing part
    params = []
    if resampleFlag == 0:
        for i in range(B):
            params.append((i, X1, hom_dim, pos2idx_X1, intervals, gen_func, gen_params))
    else:
        for i in range(B):
            params.append((i, X1, hom_dim, pos2idx_X1, intervals, resample_func, resample_params))

    # multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(loop, params[i]) for i in range(B)]
        # retrieving the output of the multiple processes
        output = []
        for f in concurrent.futures.as_completed(results):
            output.append(f.result())

    # computing total prevalence score for each individual cycle
    score = np.concatenate(output, axis=0)
    score = score.sum(axis=0)
    score_idx = np.argsort(score, kind='stable')
    score_idx = [x for x in score_idx]
    score_idx.reverse()

    # here the necessary calculations end
    end = time.time()
    total_runtime = end - start
    print("\n Total runtime = "+str(total_runtime))
    # first two most prevalent cycles
    g_ind = score_idx[0]
    g_ind2 = score_idx[1]
    if d == 3:
        plot_complex3D(X1, alpha.ST, r)
        plot_1D_generators(X1, generators[g_ind], dim)
        plot_complex3D(X1, alpha.ST, r)
        plot_1D_generators(X1, generators[g_ind2], dim)
    elif d==2:
        plot_complex(X1, alpha.ST, r)
        plot_1D_generators(X1, generators[g_ind], dim)
        plot_complex(X1, alpha.ST, r)
        plot_1D_generators(X1, generators[g_ind2], dim)
    # cycle prevalence figure
    xx = [i + 1 for i in range(len(score))]
    sortedPrevalece = [score[i] for i in score_idx]
    plt.figure()
    plt.bar(xx, sortedPrevalece, 0.9)
    plt.xticks(range(1, 21))
    plt.axis([0, 21, 0, B + 1])
    plt.title('Cycles Prevalence', fontsize='xx-large')
    plt.ylabel(' Score[%]', fontsize='xx-large')

    # persistence diagram
    b = [b for (b, d) in intervals]
    d = [d for (b, d) in intervals]
    plt.figure()
    plt.plot([0, M], [0, M], 'k--')
    plt.plot(b, d, 'r.')
    plt.axis([0, M, 0, M])
    plt.title('Persistence Diagram', fontsize='xx-large')
    plt.xlabel('birth', fontsize='xx-large')
    plt.ylabel('death', fontsize='xx-large')

    # persistence diagram log-scale
    plt.figure()
    plt.axis('scaled')
    plt.axis([log(0.0000001), log(M), log(0.0000001), log(M)])
    plt.plot([log(0.0000001), log(1)], [log(0.0000001), log(1)], 'k:', linewidth=1)
    log_b1 = [log(b) for (b, d) in intervals]
    log_d1 = [log(d) for (b, d) in intervals]
    plt.plot(log_b1, log_d1, 'r.', markersize=10)
    # plt.plot(log_b1[l], log_d1[l], 'g.', markersize=10)
    plt.title('Persistence Diagram (log-scale)', fontsize='xx-large')
    plt.xlabel('Birth', fontsize='xx-large')
    plt.ylabel('Death', fontsize='xx-large')

    plt.show()
