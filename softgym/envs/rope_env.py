import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.envs.action_space import ParallelGripper, Picker
from copy import deepcopy


class RopeEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=250, render_mode='particle', **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point_pos', 'key_point_vel', 'sphere', 'picker']
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        self.horizon = horizon

        self.cached_configs = [self.get_default_config()]
        self.cached_init_states = [None]

    def initialize_camera(self):
        pass

    def _reset(self):
        pass

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0., 4., 0.]),
                                   'angle': np.array([0, -70 / 180. * np.pi, 0.]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _get_obs(self):  # NOTE: just rename to _get_obs
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self._get_obs_key_point_idx(), :3]
        if self.observation_mode == 'point_cloud':
            pos = particle_pos
        elif self.observation_mode == 'key_point':
            pos = keypoint_pos
        elif self.observation_mode == 'cam_rgb':
            return self.render().flatten()

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim

    def _get_obs_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = self.current_config['ClothSize'][0] * (self.current_config['ClothSize'][1] - 1)
        return np.array([idx_p1, idx_p2])

    def _get_action_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = self.current_config['ClothSize'][0] * (self.current_config['ClothSize'][1] - 1)
        return np.array([idx_p1, idx_p2])

    """
    There's always the same parameters that you can set 
    """

    def set_scene(self, config, state=None):
        self.initialize_camera()
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                           *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height']])
        self.params = params  # YF NOTE: need to save the params for sampling goals
        pyflex.set_scene(12, [], 0)
        # if state is not None:
        #     self.set_state(state)
        self.current_config = deepcopy(config)

    # def get_state(self):
    #     # TODO: Xingyu: fix this before running CEM
    #     cur_state = super().get_state()
    #     return cur_state
    #
    # def set_state(self, state_dict):
    #     super().set_state(state_dict)

    def _get_info(self):
        return {}


if __name__ == '__main__':
    env = RopeEnv(observation_mode='key_point',
                  action_mode='picker',
                  num_picker=2,
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  render_mode='cloth',
                  num_variations=200,
                  use_cached_states=True,
                  deterministic=False)
    env.reset()
    for i in range(500):
        pyflex.step()
