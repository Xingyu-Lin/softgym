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

        self.initialize_camera()
        self.set_scene()
        self.get_pyflex_camera_params()
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        if device_id == -1 and 'gpu_id' in os.environ:
            device_id = int(os.environ['gpu_id'])
        self.device_id = device_id

    def get_pyflex_camera_params(self):
        '''
        get the screen width, height, camera position and camera angle.
        '''
        self.camera_params = {}
        pyflex_camera_param = pyflex.get_camera_params()
        camera_param = {'width': pyflex_camera_param[0],
                        'height': pyflex_camera_param[1],
                        'pos': np.array([pyflex_camera_param[2], pyflex_camera_param[3], pyflex_camera_param[4]]),
                        'angle': np.array([pyflex_camera_param[5], pyflex_camera_param[6], pyflex_camera_param[7]])}
        self.camera_params['default_camera'] = camera_param

    def get_camera_size(self, camera_name='default_camera'):
        return self.camera_params[camera_name]['width'], self.camera_params[camera_name]['height']

    def set_scene(self):
        ''' Set up the flex scene'''
        raise NotImplementedError

    @property
    def dt(self):
        # TODO get actual dt from the environment
        return 1 / 50.

    def render(self, mode='human'):
        if mode == 'rgb_array':
            img = pyflex.render()
            width, height = self.get_camera_size(camera_name='default_camera')
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(img[:, :, :4])
            # plt.show()
            return img
        elif mode == 'human':
            raise NotImplementedError

    def initialize_camera(self):
        '''
        This func sets the postion and angel of the camera
        camera_pos: np.ndarray (3x1). (x,y,z) coordinate of the camear.
        camera_angle: np.ndarray (3x1). (x,y,z) angle of the camera (in degree).

        Note: to set camera, you need 
        1) implement this function in your environement, set value of self.camera_pos and self.camera_angle.
        2) add the self.camera_pos and self.camera_angle to your scene parameters, and pass it when initialzing your scene.
        3) implement the CenterCamera function in your scene.h file.
        Pls see a sample usage in pour_water.py and yz_fluidshake.h

        if you do not want to set the camera, you can just not implement CenterCamera in your scene.h file, 
        and pass no camera params to your scene.
        '''
        # raise NotImplementedError

    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        phase = pyflex.get_phases()
        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos, 'phase': phase}

    def set_state(self, state_dict):
        pyflex.set_positions(state_dict['particle_pos'])
        pyflex.set_velocities(state_dict['particle_vel'])
        pyflex.set_shape_states(state_dict['shape_pos'])
        pyflex.set_phases(state_dict['phase'])

    def close(self):
        pyflex.clean()

    def get_colors(self):
        '''
        Overload the group parameters as colors also
        '''
        groups = pyflex.get_groups()
        return groups

    def set_colors(self, colors):
        pyflex.set_groups(colors)
