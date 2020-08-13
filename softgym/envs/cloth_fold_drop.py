import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from copy import deepcopy
from softgym.envs.cloth_fold import ClothFoldEnv


class ClothFoldDropEnv(ClothFoldEnv):
    def __init__(self, **kwargs):
        self.start_height = 0.8
        kwargs['cached_states_path'] = 'cloth_fold_drop_init_states.pkl'
        super().__init__(**kwargs)

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [64, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.07199, 0.94942, 1.15691]),
                                   'angle': np.array([0.633549, -0.397932, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }
        return config

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            config['target_pos'] = self._get_flat_pos()
            self._set_to_vertical(x_low=np.random.random() * 0.2 - 0.1, height_low=np.random.random() * 0.1 + 0.1)

            # Get height of the cloth without the gravity. With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()
            # cloth_height = np.linalg.norm(curr_pos[p1] - curr_pos[p2])

            curr_pos = pyflex.get_positions().reshape(-1, 4)
            curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.5, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _set_to_vertical(self, x_low, height_low):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < 0.5:
            curr_pos[:, 1] += 0.5 - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        # x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos