import numpy as np
import gym
from gym import Wrapper
from collections import namedtuple
from gym.spaces import Box
from rlpyt.spaces.box import Box
from rlpyt.spaces.composite import Composite
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.envs.base import EnvSpaces, EnvStep
import cv2 as cv
import collections
from rlpyt.utils.collections import is_namedtuple_class
import gym

OBS = namedtuple('OBS', ['pixels', 'location'])

INFO = namedtuple('INFO', ['performance', 'normalized_performance', 'total_steps'])


class MVPWrapper(object):
    def __init__(self, wrapped_env, act_null_value=0, force_float32=True):
        self._wrapped_env = wrapped_env
        action_dim = 3
        self.action_space = GymSpaceWrapper(
            space=gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32),
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )
        self.observation_space = Composite([Box(low=-np.inf, high=np.inf, shape=(64, 64, 3), dtype=np.float32),
                                            Box(np.array([-1] * 100), np.array([1] * 100), dtype=np.float32)],
                                           OBS)
        self.spaces = EnvSpaces(observation=self.observation_space, action=self.action_space)
        self._dtype = None
        self.current_location = None

    def sample_location(self, obs):
        location_orange = np.transpose(np.where(np.any(obs < 50, axis=-1)))
        location_pink = np.transpose(np.where(np.logical_and(obs[:, :, 0] > 160, obs[:, :, 1] < 180)))
        location_range = np.vstack([location_orange, location_pink])

        num_loc = np.shape(location_range)[0]
        if num_loc == 0:
            location = np.array([0., 0.])
        else:
            index = np.random.randint(num_loc)
            location = location_range[index]
            location = location / (obs.shape[0] - 1) * 2. - 1.
            location = np.array([location[1], location[0]])  # Revert location into uv coordinate
        return location

    def reset(self, **kwargs):
        image = self._wrapped_env.reset(**kwargs)
        image = np.array(cv.resize(image, (64, 64))).astype('float32')
        location = self.sample_location(image)
        self.current_location = location
        # obs = collections.OrderedDict()
        # obs['pixels'] = image
        # obs['location'] = np.tile(location, 50).reshape(-1).astype('float32')
        obs = OBS(pixels=image, location=np.tile(location, 50).reshape(-1).astype('float32'))
        return obs

    def denormalize(self, action):
        lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return scaled_action

    def step(self, action, **kwargs):
        if len(action) == 5:
            action =self.denormalize(action)
            image, reward, done, info = self._wrapped_env.step(action, **kwargs)
        else:
            action = self.denormalize(np.array([0., 0., *action]))[2:]
            image, reward, done, info = self._wrapped_env.step([*self.current_location, *action], **kwargs)
        image = np.array(cv.resize(image, (64, 64))).astype('float32')
        location = self.sample_location(image)
        self.current_location = location
        # obs = collections.OrderedDict()
        # obs['pixels'] = image
        # obs['location'] = np.tile(location, 50).reshape(-1).astype('float32')
        obs = OBS(pixels=image, location=np.tile(location, 50).reshape(-1).astype('float32'))
        return obs, reward, done, info_to_nt(info, name='INFO')

    def __getattr__(self, name):
        """ Relay unknown attribute access to the wrapped_env. """
        if name == '_wrapped_env':
            # Prevent recursive call on self._wrapped_env
            raise AttributeError('_wrapped_env not initialized yet!')
        return getattr(self._wrapped_env, name)


def info_to_nt(value, name="info"):
    if not isinstance(value, dict):
        return value
    ntc = globals()[name]
    # Disregard unrecognized keys:
    values = {k: info_to_nt(v, "_".join([name, k]))
              for k, v in value.items() if k in ntc._fields}
    # Can catch some missing values (doesn't nest):
    values.update({k: 0 for k in ntc._fields if k not in values})
    return ntc(**values)
