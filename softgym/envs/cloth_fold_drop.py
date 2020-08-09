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
                                  {'pos': np.array([-0.5, 2., 1.5]),
                                   'angle': np.array([20. / 180. * np.pi, -40. / 180. * np.pi, 0.]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}}
        }
        return config

    def generate_env_variation(self, num_variations=1, save_to_file=False, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
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
            curr_pos = pyflex.get_positions().reshape((-1, 4))

            target_pos = curr_pos.copy()[:, :3]
            target_pos[:, 1] = 0.05  # Set the cloth flatten on the ground. Assume that the particle radius is 0.05
            config['target_pos'] = target_pos

            # Get height of the cloth without the gravity. With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()
            cloth_height = np.linalg.norm(curr_pos[p1] - curr_pos[p2])

            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoints, :3]
            # pickpoint_pos[:, 1] += 1 + np.random.random(1)
            pickpoint_pos[:, 1] = cloth_height + np.random.random() / 2.
            pyflex.set_positions(curr_pos.flatten())

            # Pick up the cloth and wait to stablize
            for _ in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                curr_vel[pickpoints] = [0., 0., 0.]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            # Move the cloth to a fixed height
            step_size = 0.01
            timestep = int(abs(self.start_height - pickpoint_pos[0, 1]) / step_size)
            # print(self.start_height, pickpoint_pos[0, 1])
            # print('timestep:', timestep)
            for j in range(timestep):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                pickpoint_pos[:, 1] += np.sign(self.start_height - pickpoint_pos[0, 1]) * step_size
                # print(j, pickpoint_pos[:, 1])
                curr_pos[pickpoints, :3] = pickpoint_pos
                curr_vel[pickpoints] = [0., 0., 0.]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            # Pick up the cloth and wait to stablize
            for _ in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                curr_vel[pickpoints] = [0., 0., 0.]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            # if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
            #     curr_pos = pyflex.get_positions()
            #     self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos)
            picker_low = middle_point - [0.3, 0.5, 0.6]
            picker_high = middle_point + [0.5, 0.5, 0.6]
            # # print('picker low: {}, picker high: {}'.format(picker_low, picker_high))
            picker_low[1] = 0.1
            picker_high[1] = 1.2
            self.action_tool.update_picker_boundary(picker_low, picker_high)

        config = self.get_current_config()
        self.flat_pos = self._get_flat_pos()
        self.flat_pos += np.array([-1.2, 0., -0.55])
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()
        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors)

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        return self._get_obs()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.flat_pos[self.fold_group_b]
        curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + 1.2 * np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        if self.delta_reward:
            reward = self.prev_dist - curr_dist
            if set_prev_reward:
                self.prev_dist = curr_dist
        else:
            reward = -curr_dist
        return reward

    @property
    def performance_bound(self):
        max_dist = 1.043
        min_p = -2.2 * max_dist
        max_p = 0
        return min_p, max_p

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.flat_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        pb = self.performance_bound
        return {
            'performance': performance,
            'normalized_performance': (performance - pb[0]) / (pb[1] - pb[0]),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
        }