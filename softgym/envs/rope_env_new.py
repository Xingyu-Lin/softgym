import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.envs.action_space import ParallelGripper, Picker
from softgym.envs.robot_env import RobotBase
from copy import deepcopy
import os.path as osp

class RopeNewEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, horizon=75, render_mode='particle', picker_radius=0.02, **kwargs):
        self.render_mode = render_mode
        super().__init__(**kwargs)

        assert observation_mode in ['point_cloud', 'cam_rgb', 'key_point']
        assert action_mode in ['picker', 'sawyer', 'franka']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.num_picker = num_picker

        if action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, 
                picker_low=(-0.5, 0., -0.5), picker_high=(0.5, 0.4, 0.5))
            self.action_space = self.action_tool.action_space
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = 10 * 3
            else:
                max_particles = int(0.6 / 0.00625 * 1.1)
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

        self.horizon = horizon
        # print("init of rope new env done!")

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'init_pos': [0., 0., 0.],
            'stretchstiffness': 0.9,
            'bendingstiffness': 0.8,
            'radius': 0.025,
            'segment': 50.,
            'mass': 0.5,
            'scale': 0.5,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0, 1.3, 0]),
                                   'angle': np.array([0 * np.pi, -90 / 180. * np.pi, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_height, self.camera_width)
        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[4:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[4:, :3]
            keypoint_pos = particle_pos[self.key_point_indices, :3]
            pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def _get_key_point_idx(self, num=None):
        indices = [0, num - 1]
        interval = (num - 2) // 8
        for i in range(1, 9):
            indices.append(i * interval)

        return indices

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        else:
            render_mode = 2

        camera_params = config['camera_params'][config['camera_name']]
        params = np.array(
            [*config['init_pos'], config['stretchstiffness'], config['bendingstiffness'], config['radius'], config['segment'], config['mass'], 
                config['scale'], 
                *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], render_mode]
            )

        # print("right before pyflex set_scene!")
        if self.version == 2:
            pyflex.set_scene(16, params, 0, [0])
        elif self.version == 1:
            pyflex.set_scene(16, params, 0)
        
        num_particles = pyflex.get_n_particles()
        # print("with {} segments, the number of particles are {}".format(config['segment'], num_particles))
        # exit()
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

    def _center_object(self):
        """ 
        Center the object to be at the origin
        NOTE: call a pyflex.set_positions and then pyflex.step
        """
        pos = pyflex.get_positions().reshape(-1, self.dim_position)
        pos[4:, [0, 2]] -= np.mean(pos[4:, [0, 2]], axis=0, keepdims=True)
        
        pyflex.set_positions(pos.flatten())
        pyflex.step()
        pyflex.render()



if __name__ == '__main__':
    env = RopeNewEnv(observation_mode='key_point',
                  action_mode='picker',
                  num_picker=2,
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=10,
                  use_cached_states=False,
                  save_cache_states=False,
                  deterministic=False)
    env.reset(config=env.get_default_config())
    for i in range(1000):
        print(i)
        print("right before pyflex step")
        pyflex.step()
        print("right after pyflex step")
        print("right before pyflex render")
        pyflex.render()
        print("right after pyflex render")
