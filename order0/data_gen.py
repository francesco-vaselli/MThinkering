import numpy as np
from numpy.random import default_rng


if __name__ == '__main__':

    rng = default_rng(42)

    dataset_size = 5e6
    dataset = []

    for i in range(0, int(dataset_size)):

        gen = rng.integers(5, 11)
        # gen = 10
        mu, sigma = gen, gen*0.1 # mean and standard deviation
        reco = rng.normal(mu, sigma)

        dataset.append([gen, reco])
    
    dataset = np.array(dataset, dtype=np.float32)
    np.save('order0/dataset.npy', dataset)