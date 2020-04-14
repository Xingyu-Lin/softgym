import numpy as np

class RandomPolicy:

    def __init__(self, n_acts):
        self._n_acts = n_acts

    def __call__(self, obs):
        n_envs = obs.shape[0]
        return np.random.random((n_envs, self._n_acts)) * 2 - 1