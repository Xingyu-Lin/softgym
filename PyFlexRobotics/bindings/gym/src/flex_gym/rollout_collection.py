import logging
from time import time
import numpy as np

def _increment_mean(mean, N, x, M):
    return (N * mean + M * x) / (N + M)

class _RolloutBuffer:

    def __init__(self, n, nenvs):
        self._n = n
        self._nenvs = nenvs
        self.reset()        

    def append(self, obs, acts, rews, dones, infos):
        if self.full:
            raise ValueError('Buffer capacity reached! Can no longer call append!')
        for i in range(self._nenvs):
            self._tmp_data['obs'][i].append(obs[i].copy())
            self._tmp_data['acts'][i].append(acts[i].copy())
            self._tmp_data['rews'][i].append(rews[i].copy())
            self._tmp_data['dones'][i].append(dones[i].copy())
            self._tmp_data['infos'][i].append(infos[i].copy())

            if dones[i]:
                for key, val in self._data.items():
                    val[self.size] = self._tmp_data[key][i]
                    self._tmp_data[key][i] = []
                self._size += 1

                if self.full:
                    break

    @property
    def full(self):
        return self._size >= self._n

    @property
    def size(self):
        return self._size

    def reset(self):        
        self._data = {
            'obs': [[] for _ in range(self._n)],
            'acts': [[] for _ in range(self._n)],
            'rews': [[] for _ in range(self._n)],
            'dones': [[] for _ in range(self._n)],
            'infos': [[] for _ in range(self._n)],
        }
        self._tmp_data = {key: [[] for _ in range(self._nenvs)] for key in self._data.keys()}
        self._size = 0

    def flush(self):
        for val in self._data.values():
            for i in range(self._n):
                val[i] = np.array(val[i])
        data = self._data
        self.reset()
        return data

class RolloutCollector:

    def __init__(self, env, policy):
        self._env = env
        self._policy = policy

    def get_rollouts(self, n):
        buffer = _RolloutBuffer(n, self._env.num_envs)
        
        obs = self._env.reset(np.arange(self._env.num_envs))
        t_start = time()
        t_last = time()
        t_mean = 0
        size_last = buffer.size
        while not buffer.full:
            acts = self._policy(obs)
            obs_t1, rews, dones, infos = self._env.step(acts)
            buffer.append(obs, acts, rews, dones, infos)
            obs = obs_t1
            self._env.reset()

            if buffer.size != size_last:
                t_cur = time()
                t_mean = _increment_mean(t_mean, size_last, t_cur - t_last, buffer.size - size_last)
                logging.info('Collected {}/{}. Took {:.2f}s. Est {:.2f}s'.format(
                    buffer.size, n, time() - t_start, t_mean * (n - buffer.size)
                ))
                size_last = buffer.size
                t_last = t_cur

        return buffer.flush()

        

            


