import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.envs.action_space import ParallelGripper, Picker, PickerPickPlace
from copy import deepcopy


class ClothEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, render_mode='particle', picker_radius=0.07, particle_radius=0.05, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['sphere', 'picker', 'pickerpickplace']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.cloth_particle_radius = particle_radius

        if action_mode.startswith('key_point'):
            space_low = np.array([0, -0.1, -0.1, -0.1] * 2)
            space_high = np.array([3.9, 0.1, 0.1, 0.1] * 2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode.startswith('sphere'):
            self.action_tool = ParallelGripper(gripper_type='sphere')
            self.action_space = self.action_tool.action_space
        elif action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius)
            self.action_space = self.action_tool.action_space
        elif action_mode == 'pickerpickplace':
            self.action_tool = PickerPickPlace(num_picker=num_picker, particle_radius=particle_radius, env=self)
            self.action_space = self.action_tool.action_space

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = len(self._get_key_point_idx()) * 3
            else:
                max_particles = 64 * 40
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode in ['picker']:
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

    def _sample_cloth_size(self):
        return np.random.randint(10, 64), np.random.randint(10, 40)

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear, Before ICML rebuttal
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
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    def _get_key_point_idx(self):
        """ The keypoints are defined as the four corner points of the cloth """
        dimx, dimy = self.current_config['ClothSize']
        idx_p1 = 0
        idx_p2 = dimx * (dimy - 1)
        idx_p3 = dimx - 1
        idx_p4 = dimx * dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4])

    """
    There's always the same parameters that you can set 
    """

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                           *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height']])
        env_idx = 9 if 'env_idx' not in config else config['env_idx']
        self.params = params  # YF NOTE: need to save the params for sampling goals
        pyflex.set_scene(env_idx, params, 0)
        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    # def get_state(self):
    #     # TODO: Xingyu: fix this before running CEM
    #     cur_state = super().get_state()
    #     return cur_state
    #
    # def set_state(self, state_dict):
    #     super().set_state(state_dict)
