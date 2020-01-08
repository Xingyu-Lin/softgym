import os
import yaml
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import gym
from softgym.utils.visualization import save_numpy_as_gif
import cv2
import os.path as osp

try:
    import pyflex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (You need to first compile the python binding)".format(e))


class FlexEnv(gym.Env):
    def __init__(self, device_id=-1, headless=False, render=True, horizon=100, camera_width=720, camera_height=720, action_repeat=8,
                 camera_name='default_camera'):
        self.camera_width, self.camera_height, self.camera_name = camera_width, camera_height, camera_name
        pyflex.init(headless, render, camera_width,
                    camera_height)  # TODO check if pyflex needs to be initialized for each instance of the environment
        self.record_video, self.video_path, self.video_name = False, None, None

        self.set_scene()
        # self.set_video_recording_params()
        self.get_pyflex_camera_params()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        if device_id == -1 and 'gpu_id' in os.environ:
            device_id = int(os.environ['gpu_id'])
        self.device_id = device_id

        self.horizon = horizon
        self.time_step = 0
        self.action_repeat = action_repeat
        self.recording = False

    @staticmethod
    def _load_config(config_name):
        """ Assume that the .yaml config file is under the same directory as the env files """
        config_dir = osp.dirname(osp.abspath(__file__))
        config_stream = open(osp.join(config_dir, config_name), 'r').read()
        return yaml.load(config_stream, Loader=yaml.FullLoader)

    def get_pyflex_camera_params(self):
        """ get the screen width, height, camera position and camera angle. """
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
        """ Set up the flex scene """
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
            return img
        elif mode == 'human':
            raise NotImplementedError

    def initialize_camera(self):
        '''
        This function sets the postion and angel of the camera
        camera_pos: np.ndarray (3x1). (x,y,z) coordinate of the camera
        camera_angle: np.ndarray (3x1). (x,y,z) angle of the camera (in degree).

        Note: to set camera, you need 
        1) implement this function in your environement, set value of self.camera_pos and self.camera_angle.
        2) add the self.camera_pos and self.camera_angle to your scene parameters,
            and pass it when initializing your scene.
        3) implement the CenterCamera function in your scene.h file.
        Pls see a sample usage in pour_water.py and softgym_PourWater.h

        if you do not want to set the camera, you can just not implement CenterCamera in your scene.h file, 
        and pass no camera params to your scene.
        '''
        raise NotImplementedError

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

    def start_record(self):
        self.video_frames = []
        self.recording = True

    def end_record(self, video_path, **kwargs):
        if not self.recording:
            print('function end_record: Error! Not recording video')
        self.recording = False
        save_numpy_as_gif(np.array(self.video_frames), video_path, **kwargs)
        del self.video_frames

    def reset(self):
        self._reset()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))

    def step(self, action):
        next_state, reward, done, info = self._step(action)
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.time_step += 1

        done = False
        if self.time_step == self.horizon:
            done = True
        return next_state, reward, done, info

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _seed(self):
        pass

    def get_image(self, width=960, height=720):
        '''
        use pyflex.render to get a rendered image.
        '''
        img = pyflex.render()
        img = img.reshape(self.camera_height, self.camera_width, 4)[::-1, :, :3]  # Need to reverse the height dimension
        img = img.astype(np.uint8)
        # img = img[:,:,::-1]
        # cv2.imshow('ImageEnv', img)
        # cv2.waitKey(0)
        # if self.time_step  == 200:
        #     print("show image")
        #     plt.imshow(img)
        #     plt.show()
        img = cv2.resize(img, (width, height))  # add this to align with img env. TODO: this seems to have some problems.
        # img = img.reshape((width, height, 3)) # in pytorch format, to algin with imgenv
        # cv2.imshow('ImageEnv2', img)
        # cv2.waitKey(0)
        return img

    def close(self):
        pyflex.clean()
