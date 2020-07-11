from abc import ABC

import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.envs.action_space import ParallelGripper, Picker, PickerPickPlace
from softgym.envs.robot_env import RobotBase
from copy import deepcopy


class RigidClothEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_pieces, num_picker=2, picker_radius=0.05, particle_radius=0.02, **kwargs):
        self.cloth_particle_radius = particle_radius
        super().__init__(**kwargs)

        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['sphere', 'picker', 'pickerpickplace', 'sawyer', 'franka']
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.num_pieces = num_pieces

        if action_mode.startswith('key_point'):
            space_low = np.array([0, -0.1, -0.1, -0.1] * 2)
            space_high = np.array([3.9, 0.1, 0.1, 0.1] * 2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode.startswith('sphere'):
            self.action_tool = ParallelGripper(gripper_type='sphere')
            self.action_space = self.action_tool.action_space
        elif action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius,
                                      picker_low=(-0.4, 0., -0.4), picker_high=(1.0, 0.5, 0.4))

            # for i in range(100):
            #     pyflex.step(render=True)
            self.action_space = self.action_tool.action_space
        elif action_mode == 'pickerpickplace':
            self.action_tool = PickerPickPlace(num_picker=num_picker, particle_radius=particle_radius, env=self)
            self.action_space = self.action_tool.action_space
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)

        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = len(self._get_key_point_idx()) * 3
            else:
                max_particles = 500 * num_pieces
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
        """ Size of just one piece"""
        return np.random.randint(8, 15), np.random.randint(8, 20)

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothSize': [20, 20],  # Size of one piece
            'camera_name': 'default_camera',
            'inv_mass': 0.001,
            'rigid_stiffness': 1,
            'num_pieces': 2,
            'camera_params': {'default_camera':
                                  {'pos': np.array([-0.0, 0.82, 0.82]),
                                   'angle': np.array([0, -45 / 180. * np.pi, 0.]),
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

    def _get_key_point_idx(self):
        """ The keypoints are defined as the four corner points of the cloth """
        dimx, dimy = self.current_config['ClothSize']
        piece_size = dimx * dimy
        idx_p1 = 0
        idx_p2 = dimy - 1
        idx_p3 = piece_size - dimy
        idx_p4 = piece_size - 1
        points1 = np.array([idx_p1, idx_p2, idx_p3, idx_p4])
        if self.num_pieces == 1:
            return points1
        elif self.num_pieces == 2:
            points2 = points1.copy() + piece_size
            return np.concatenate([points1, points2])
        else:
            raise NotImplementedError

    """
    There's always the same parameters that you can set 
    """

    def set_scene(self, config, state=None):
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 14
        scene_params = np.array(
            [config['ClothSize'][0], 1, config['ClothSize'][1], config['num_pieces'], config['inv_mass'], config['rigid_stiffness'],
             *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height']])

        if self.version == 2:
            robot_params = [0]
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)

        self.current_config = deepcopy(config)
