from typing import BinaryIO
import numpy as np
from statistics import median
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from numba import njit
import timeit


# @njit
def bitter(data, bin_exp):
    """function returning series of bits according to position in histogram with 2**bin_exp bins.
    the array gets split into two equal widths bins and we assing 0 to values to the left and 1 to values to the right.
    at each subsequent iteration each bin is split into two other equal widths bins and the same assignment is performed for each value

    if calling just once, bare numpy is faster. for multiple calls numba provides 2x speedup; however np.histogram has to be used instead

    Args:
        data ([ 1d np.array]): input dataset
        bin_exp ([int]): parameter setting the final number of bins as n = 2**bin_exp

    Returns:
        [(len(data), bin_exp) np.array]: bitted array in the same ordering as data
    """    
    # _, 
    bin_edges = np.histogram_bin_edges(data, bins=2)
    inds = np.digitize(data, bin_edges)
    bits = np.where(inds%2 == 1, 0, 1).reshape(-1, 1)

    for j in range(2, bin_exp+1):
        # _, 
        bin_edges = np.histogram_bin_edges(data, bins=2**j)
        # print(bin_edges)
        inds = np.digitize(data, bin_edges)
        next_bit = np.where(inds%2 == 1, 0, 1).reshape(-1, 1)
        bits = np.hstack((bits, next_bit))

    return bin_edges, bits


def debitter(bits, bin_edges):

    bins = []
    for _, bit in enumerate(bits):
        if bit[0] == 1:
            bin = 2
        else:
            bin = 1

        for j in range(1, len(bit)):
            if bit[j] == 1:
                bin += bin
            else:
                bin += (bin -1)
        
        bins.append(bin)
    
    bins = np.array(bins)
    hist_min = bin_edges.min()
    print(hist_min)
    bin_width = np.ediff1d(bin_edges)[0]
    hist_centre = lambda t: hist_min + bin_width*(t-0.5)
    histed = np.vectorize(hist_centre)(bins)
    
    return histed


if __name__ == '__main__':
    rng = default_rng(43)

    BATCH_SIZE = 1024
    BUFFER_SIZE = BATCH_SIZE*10
    noise_dim = 16

    dataset_g = rng.normal(10, 1, int(BUFFER_SIZE/2))
    full = np.array(dataset_g, dtype=np.float32)
    
    plt.hist(full, bins=2**7)

    # t = timeit.Timer(lambda: bitter(full, 10)) 
    # print (t.timeit(1))

    bin_edges, bitted = bitter(full, 7)
    print(full[10], bitted[10])
    histed = debitter(bitted, bin_edges)
    f = plt.figure()
    plt.hist(histed, bins=bin_edges)

    plt.show()   
