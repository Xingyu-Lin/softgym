import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import pyflex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (You need to first compile the python binding)".format(e))


class FlexEnv(gym.Env):
    def __init__(self, device_id=-1):
        pyflex.init()  # TODO check if pyflex needs to be initialized for each instance of the environment
        self.set_scene()
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        if device_id == -1 and 'gpu_id' in os.environ:
            device_id = int(os.environ['gpu_id'])
        self.device_id = device_id

    def set_scene(self):
        ''' Set up the flex scene'''
        raise NotImplementedError

    @property
    def dt(self):
        # TODO get actual dt from the environment
        return 1/50.

    def render(self, mode='human'):
        if mode == 'rgb_array':
            raise NotImplementedError
        elif mode == 'human':
            raise NotImplementedError

    def initialize_camera(self):
        raise NotImplementedError

    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos}

    def set_state(self, state_dict):
        pyflex.set_positions(state_dict['particle_pos'])
        pyflex.set_velocities(state_dict['particle_vel'])
        pyflex.set_shape_states(state_dict['shape_pos'])

    def close(self):
        pyflex.clean()