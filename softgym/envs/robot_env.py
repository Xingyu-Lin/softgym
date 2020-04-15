import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.envs.action_space import ParallelGripper, Picker, PickerPickPlace
from copy import deepcopy
import pickle


class RobotEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, render_mode='particle', picker_radius=0.07,
                 cached_states_path='robot_init_states.pkl', **kwargs):
        self.render_mode = render_mode
        self.cached_states_path = cached_states_path
        super().__init__(**kwargs)

        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['sphere', 'picker', 'pickerpickplace']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.cloth_particle_radius = 0.05  # Hardcoded radius

        space_low = np.array([0, -0.1, -0.1, -0.1] * 2)
        space_high = np.array([3.9, 0.1, 0.1, 0.1] * 2)
        self.action_space = Box(space_low, space_high, dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                     dtype=np.float32)
        success = self.get_cached_configs_and_states(cached_states_path)

        if not success or not self.use_cached_states:
            self.cached_configs, self.cached_init_states = self.generate_env_variation(self.num_variations, save_to_file=True)
            success = self.get_cached_configs_and_states(cached_states_path)
            assert success

    def generate_env_variation(self, num_variations, save_to_file=False, **kwargs):
        default_config = self.get_default_config()
        generated_configs, generated_states = [], []

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            self.set_scene(config)
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return generated_configs, generated_states

    def _step(self, action):
        pass

    def _reset(self):
        return self._get_obs()

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0, 4., 1]),
                                   'angle': np.array([0, -70 / 180. * np.pi, 0.]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        else:
            raise NotImplementedError

    """
    There's always the same parameters that you can set 
    """

    def set_scene(self, config, state=None):
        params = []
        pyflex.set_scene(13, params, 0)
        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        return 0.

    def _get_info(self):
        return {}
