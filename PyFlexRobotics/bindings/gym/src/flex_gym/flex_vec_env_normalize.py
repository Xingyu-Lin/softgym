import numpy as np
from baselines.common.running_mean_std import RunningMeanStd

from flex_vec_env import FlexVecEnv

class FlexVecEnvNormalize(FlexVecEnv):

    def __init__(self, flex_env_cfg, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super().__init__(flex_env_cfg)
        self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, vac):
        obs, rews, news, infos = super().step(vac)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self, init=True):
        obs = super().reset(init=init)
        return self._obfilt(obs)
