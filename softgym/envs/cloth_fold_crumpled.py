import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from copy import deepcopy
from softgym.envs.cloth_fold import ClothFoldEnv


class ClothFoldCrumpledEnv(ClothFoldEnv):
    def __init__(self, **kwargs):
        kwargs['cached_states_path'] = 'cloth_fold_crumpled_init_states.pkl'
        super().__init__(**kwargs)

    def generate_env_variation(self, num_variations=2, save_to_file=False, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.001  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            print('generating variation {}'.format(i))
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            num_particle = cloth_dimx * cloth_dimy
            pickpoint = random.randint(0, num_particle - 1)
            curr_pos = pyflex.get_positions()
            original_inv_mass = curr_pos[pickpoint * 4 + 3]
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
            pickpoint_pos[1] += np.random.random(1) * 0.5
            pyflex.set_positions(curr_pos)

            # Pick up the cloth and wait to stablize
            for _ in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            # Drop the cloth and wait to stablize
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = original_inv_mass
            pyflex.set_positions(curr_pos)
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break

            self._center_object()

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        # if hasattr(self, 'action_tool'):
        #     x = pyflex.get_positions().reshape((-1, 4))[0][0]  # x coordinate of left-top corner
        #     self.action_tool.reset([x + 0.1, 0.2, 0])
        #     picker_low = self.action_tool.picker_low
        #     picker_high = self.action_tool.picker_high
        #     offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
        #     picker_low[0] += offset_x
        #     picker_high[0] += offset_x
        #     picker_high[0] += 1.0
        #     self.action_tool.update_picker_boundary(picker_low, picker_high)

        if hasattr(self, 'action_tool'):
            # curr_pos = pyflex.get_positions()
            self.action_tool.reset([0., 0.5, 0.])
            self.action_tool.update_picker_boundary(picker_low=[-0.5, 0., -0.5], picker_high=[1.5, 2., 1.5])

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        self.set_colors(colors)

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
        pos_group_b_init = self.init_pos[self.fold_group_b]
        curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + \
                    0.2 * np.linalg.norm(np.mean(pos_group_b, axis=0) - np.mean(pos_group_b_init, axis=0))
        if self.delta_reward:
            reward = self.prev_dist - curr_dist
            if set_prev_reward:
                self.prev_dist = curr_dist
        else:
            reward = -curr_dist
        return reward