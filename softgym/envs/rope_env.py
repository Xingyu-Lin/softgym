import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.envs.action_space import ParallelGripper, Picker
from copy import deepcopy


class RopeEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=250, render_mode='particle', picker_radius=0.15, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb', 'key_point']
        assert action_mode in ['picker']
        self.observation_mode = observation_mode
        self.action_mode = action_mode

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, picker_low=(-1.5, 0., -1.), picker_high=(4.5, 2.8, 4.))
            self.action_space = self.action_tool.action_space

        max_particles = 30
        if observation_mode in ['key_point']:
            obs_dim = len(self._get_key_point_idx())
            if action_mode in ['picker']:
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

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
            'ParticleInvMass': 0.01,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0., 7., 3.]),
                                   'angle': np.array([0, -65 / 180. * np.pi, 0.]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            pos = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def _get_key_point_idx(self):
        """ Return the two endpoints on the rope as the keypoints """
        idx_p1 = 0
        idx_p2 = 159  # Hardcode the keypoint index as the rope does not change
        return np.array([idx_p1, idx_p2])

    """
    There's always the same parameters that you can set 
    """

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2
        params = np.array(
            [5, config['ClusterSpacing'], config['ClusterRadius'], config['ClusterStiffness'], config['DynamicFriction'], config['ParticleFriction']])
        pyflex.set_scene(12, params, 0)
        self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def _get_info(self):
        return {}

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)


if __name__ == '__main__':
    env = RopeEnv(observation_mode='point_cloud',
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
