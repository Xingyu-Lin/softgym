import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from copy import deepcopy
from os import path as osp

class DoughEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, horizon=75, cached_states_path='dough_flatten_init_states.pkl', 
            render_mode='particle', **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb']
        assert action_mode in ['direct'] # direct control the capsule
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        max_particles = 100
        if observation_mode == 'point_cloud':
            # all particle positions + capsule pos & quat
            self.observation_space = Box(np.array([-np.inf] * (max_particles * 3 + 7)),
                                         np.array([np.inf] * (max_particles * 3 + 7)), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        self.action_space = Box(np.array([-0.2] * 4), np.array([0.2] * 4), dtype=np.float32)
        self.horizon = horizon

        self.cached_configs = [self.get_default_config()]
        self.cached_init_states = [None]

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClusterSpacing': 1.5,
            'ClusterRadius': 0.,
            'ClusterStiffness': 0.55,
            'DynamicFriction': 3.0,
            'ParticleFriction': 0.25,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0., 7., 3.]),
                                   'angle': np.array([0, -65 / 180. * np.pi, 0.]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _reset(self):
        return self._get_obs()

    def _get_obs(self):
        if self.observation_mode == 'full_state':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            return particle_pos
        elif self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)

    def set_scene(self, config, state=None):
        # TODO: pass render mode into the scene
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2

        # add a sphere plastic dough 
        params = np.array(
            [config['ClusterSpacing'], config['ClusterRadius'], config['ClusterStiffness'], config['DynamicFriction'], config['ParticleFriction']])
        pyflex.set_scene(13, params, 0)
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

        if state is not None:
            self.set_state(state)
        else:
            for _ in range(300): # wait for dough to stablize
                pyflex.step()

    def _get_info(self):
        return {}

if __name__ == '__main__':
    env = DoughEnv(observation_mode='full_state',
                  action_mode='direct',
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=200,
                  use_cached_states=True,
                  deterministic=False)

    env.reset()
    for i in range(500):
        pyflex.step()
