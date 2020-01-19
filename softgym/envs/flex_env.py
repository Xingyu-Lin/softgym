import os
import copy
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import gym
from softgym.utils.visualization import save_numpy_as_gif
import cv2
import os.path as osp
import pickle

try:
    import pyflex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (You need to first compile the python binding)".format(e))


class FlexEnv(gym.Env):
    def __init__(self, device_id=-1, headless=False, render=True, horizon=100, camera_width=720, camera_height=720,
                 action_repeat=8, camera_name='default_camera', delta_reward=True, deterministic=True, use_cached_states=True, **kwargs):
        self.camera_params, self.camera_width, self.camera_height, self.camera_name = {}, camera_width, camera_height, camera_name
        pyflex.init(headless, render, camera_width, camera_height)
        self.record_video, self.video_path, self.video_name = False, None, None

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
        self.prev_reward = None
        self.delta_reward = delta_reward
        self.deterministic = deterministic
        self.use_cached_states = use_cached_states
        self.current_config = self.get_default_config()
        self.cached_configs, self.cached_init_states = None, None

        self.dim_position = 4
        self.dim_velocity = 3
        self.dim_shape_state = 14
        self.particle_num = 0
        
    def _center_object(self):
        """ 
        Center the object to be at the origin
        NOTE: call a pyflex.set_positions and then pyflex.step
        """
        pos = pyflex.get_positions().reshape(-1, self.dim_position)
        pos[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
        pyflex.set_positions(pos.flatten())
        pyflex.step()
        import time
        time.sleep(5)

    def get_cached_configs_and_states(self, cached_states_path):
        """
        If the path exists, load from it. Should be a list of (config, states)
        :param cached_states_path:
        :return:
        """
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            cached_states_path = osp.join(cur_dir, cached_states_path)
        if not osp.exists(cached_states_path):
            return False
        with open(cached_states_path, "rb") as handle:
            self.cached_configs, self.cached_init_states = pickle.load(handle)
        print('{} config and state pairs loaded from {}'.format(len(self.cached_init_states), cached_states_path))
        return True

    def get_default_config(self):
        """ Generate the default config of the environment scenes"""
        raise NotImplementedError

    def generate_env_variation(self, num_variations, save_to_file=False, **kwargs):
        """
        Generate a list of configs and states
        :return:
        """
        raise NotImplementedError

    def get_current_config(self):
        return self.current_config

    def get_camera_size(self, camera_name='default_camera'):
        return self.camera_params[camera_name]['width'], self.camera_params[camera_name]['height']

    def update_camera(self, camera_name, camera_param=None):
        """
        :param camera_name: The camera_name to switch to
        :param camera_param: None if only switching cameras. Otherwise, should be a dictionary
        :return:
        """
        if camera_param is not None:
            self.camera_params[camera_name] = camera_param
        else:
            camera_param = self.camera_params[camera_name]
        pyflex.set_camera_params(
            np.array([*camera_param['pos'], *camera_param['angle'], camera_param['width'], camera_param['height']]))

    def set_scene(self, config, state=None):
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
        """
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
        """
        raise NotImplementedError

    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        phase = pyflex.get_phases()
        camera_params = copy.deepcopy(self.camera_params)

        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos, 'phase': phase, 'camera_params': camera_params}

    def set_state(self, state_dict):
        pyflex.set_positions(state_dict['particle_pos'])
        pyflex.set_velocities(state_dict['particle_vel'])
        pyflex.set_shape_states(state_dict['shape_pos'])
        pyflex.set_phases(state_dict['phase'])
        self.camera_params = copy.deepcopy(state_dict['camera_params'])
        self.update_camera(self.camera_name)

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
        config_id = np.random.randint(len(self.cached_configs)) if not self.deterministic else 0
        self.current_config = self.cached_configs[config_id]
        self.current_config_idx = config_id
        self.set_scene(self.cached_configs[config_id], self.cached_init_states[config_id])

        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0 
        obs = self._reset()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        return obs

    def step(self, action):
        for _ in range(self.action_repeat):
            self._step(action)
        obs = self._get_obs()
        reward = self.compute_reward(action, obs, set_prev_reward=True)
        info = self._get_info()

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.time_step += 1

        done = False
        if self.time_step == self.horizon:
            done = True

        return obs, reward, done, info

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ set_prev_reward is used for calculate delta rewards"""
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _seed(self):
        pass

    def get_image(self, width=960, height=720):
        '''
        use pyflex.render to get a rendered image.
        this is in support for the multitask env.
        '''
        # raise DeprecationWarning
        # img = pyflex.render()
        # img = img.reshape(self.camera_height, self.camera_width, 4)[::-1, :, :3]  # Need to reverse the height dimension
        # img = img.astype(np.uint8)
        # img = cv2.resize(img, (width, height))  # add this to align with img env. TODO: this seems to have some problems.
        # return img
        img = self.render(mode='rgb_array')
        img = img.astype(np.uint8)
        img = cv2.resize(img, (width, height))
        return img