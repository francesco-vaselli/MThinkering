from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

class DNN_gen(stats.rv_continuous):

    def _pdf(self, x):
        rv = stats.beta(10, 1)
        rvm = stats.beta(10, 1, loc=1)
        return 1/2*(rv.pdf(x) + (1 - rvm.pdf(x)))



if __name__ == '__main__':

    rng = default_rng(43)
    dataset = np.array(rng.normal(10, 1, 1000), dtype=np.float32)

    # dist = DNN_gen(name='dist')
    '''
    list = []
    for i in range(0, 100):
        x = np.random.uniform(0, 1)
        l_r = (stats.beta.rvs(x, 10,0))
        l_l = (2 - stats.beta.rvs(x, 10,1))
        # print(x, l_r+l_l)
        list.append(l_r)
        list.append(l_l)
    '''
    list = []
    b1 = (1 - stats.beta.rvs(a=17, b=1, size=500))
    b2 = (stats.beta.rvs(a=17, b=1, size=500))
    data_beta_a10b1 = np.concatenate((b1, b2))
    rng.shuffle(data_beta_a10b1)
    
    full = np.column_stack((dataset, data_beta_a10b1))
    print(full)
    plt.plot(full[:, 0], full[:, 1], 'o', markersize=1)
    plt.show()

    print(len(data_beta_a10b1))
    plt.hist(data_beta_a10b1, bins=50, density=True, alpha=0.6)
    # plt.show()

