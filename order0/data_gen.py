import numpy as np


if __name__ == '__main__':


    N = np.random.randint(1, 7)

    real_event = np.random.default_rng().uniform(low=5, high=100, size=(1, N))
    real_event = np.sort(real_event)
    print(len(real_event[0]))

    real_event
