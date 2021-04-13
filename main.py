import numpy as np
import matplotlib.pyplot as plt
from coupled_alpha import CoupledAlpha
from alpha_complex import AlphaComplex
from utils import *
import math
import concurrent.futures



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
    return gentorus(gen_params)


if __name__ == '__main__':
    # number of points in a sample
    N = 100
    # number of resamples
    B = 100
    # dimension of homology
    hom_dim = 1
    # gen_func parameters
    r1 = 0.5
    r2 = 0.2
    gen_params = [r1, r2, N]
    # filtration value for figures
    r = 0.25
    r = r ** 2
    # persistence diagram figure upper limit
    M = 0.3*r1

    X1 = gen_func(gen_params)
    dim = X1.shape[1]
    alpha = AlphaComplex(X1)
    intervals, generators, _, pos2idx_X1 = alpha.persistent_homology_in_dimension(hom_dim)

    # list of params for the multiprocessing part
    params = []
    for i in range(B):
        params.append((i, X1, hom_dim, pos2idx_X1, intervals, gen_func, gen_params))

    # multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(loop, params[i]) for i in range(B)]
        # retrieving the output of the multiple processes
        output = []
        for f in concurrent.futures.as_completed(results):
            output.append(f.result())

    # computing total score for each individual cycle
    score = np.concatenate(output, axis=0)
    score = score.sum(axis=0)
    score_idx = np.argsort(score, kind='stable')
    score_idx = [x for x in score_idx]
    score_idx.reverse()

    # first two most prevalent cycles
    g_ind = score_idx[0]
    g_ind2 = score_idx[1]
    plot_complex3D(X1, alpha.ST, r)
    plot_1D_generators(X1, generators[g_ind], dim)
    plot_complex3D(X1, alpha.ST, r)
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

    plt.show()
