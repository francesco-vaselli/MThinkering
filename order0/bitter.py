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

    if calling just once, bare numpy is faster. for multiple calls, numba provides 2x speedup

    Args:
        data ([ 1d np.array]): input dataset
        bin_exp ([int]): parameter setting the final number of bins as n = 2**bin_exp

    Returns:
        [(len(data), bin_exp) np.array]: bitted array in the same ordering as data
    """    
    _, bin_edges = np.histogram(data, bins=2)
    inds = np.digitize(data, bin_edges)
    bits = np.where(inds%2 == 1, 0, 1).reshape(-1, 1)

    for j in range(2, bin_exp+1):
        _, bin_edges = np.histogram(data, bins=2**j)
        # print(bin_edges)
        inds = np.digitize(data, bin_edges)
        next_bit = np.where(inds%2 == 1, 0, 1).reshape(-1, 1)
        bits = np.hstack((bits, next_bit))

    return bits
        


if __name__ == '__main__':
    rng = default_rng(43)

    BATCH_SIZE = 1024
    BUFFER_SIZE = BATCH_SIZE*10
    noise_dim = 16

    dataset_g = rng.normal(10, 1, int(BUFFER_SIZE/2))
    full = np.array(dataset_g, dtype=np.float32)
    print((full.max()+full.min())/2, median(full))
    # plt.hist(full, bins=100)
    # plt.show()
    t = timeit.Timer(lambda: bitter(full, 7)) 
    print (t.timeit(1))

    bitted = bitter(full, 3)
    print(full[10], bitted[10])
