import numpy as np
import gym.spaces
from gym.spaces.box import Box
from softgym.utils.overrides import overrides


class NormalizedEnv(object):
    def __init__(
      self,
      env,
      scale_reward=1.,
      normalize_obs=False,
      normalize_reward=False,
      obs_alpha=0.001,
      reward_alpha=0.001,
      clip=True,
      clip_obs=None
    ):
        self._wrapped_env = env
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.shape)
        self._obs_var = np.ones(env.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.
        self._clip = clip
        self._clip_obs = clip_obs

    def _update_obs_estimate(self, obs):
        flat_obs = self._wrapped_env.observation_space.flatten(obs)
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var + self._reward_alpha * np.square(reward -
                                                                                                        self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        ret = self._wrapped_env.reset(**kwargs)
        if self._clip_obs is not None:
            ret = np.clip(ret, self._clip_obs[0], self._clip_obs[1])
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    @property
    @overrides
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            return gym.spaces.Box(-1 * ub, ub)
        return self._wrapped_env.action_space

    def denormalize(self, action):
        lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return scaled_action

    @overrides
    def step(self, action, **kwargs):
        if isinstance(self._wrapped_env.action_space, Box):
            # rescale the action
            lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            if self._clip:
                scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        wrapped_step = self._wrapped_env.step(scaled_action, **kwargs)
        next_obs, reward, done, info = wrapped_step
        if self._clip_obs is not None:
            next_obs = np.clip(next_obs, self._clip_obs[0], self._clip_obs[1])

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        return next_obs, reward * self._scale_reward, done, info

    def __getattr__(self, name):
        """ Relay unknown attribute access to the wrapped_env. """
        if name == '_wrapped_env':
            # Prevent recursive call on self._wrapped_env
            raise AttributeError('_wrapped_env not initialized yet!')
        return getattr(self._wrapped_env, name)

    def get_model_action(self, action, curr_pos, particle_pos):
        lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        self._wrapped_env.get_model_action(scaled_action, curr_pos, particle_pos)

normalize = NormalizedEnv
