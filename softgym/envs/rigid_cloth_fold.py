import numpy as np
import pickle
import os.path as osp
import pyflex
from copy import deepcopy
from softgym.envs.rigid_cloth_env import RigidClothEnv
from softgym.utils.pyflex_utils import center_object


class RigidClothFoldEnv(RigidClothEnv):
    def __init__(self, cached_states_path='rigid_cloth_fold_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _sample_cloth_size(self):
        """ Size of just one piece"""
        return np.random.randint(8, 15), np.random.randint(10, 20)

    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
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

            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()

            keypoints = self._get_key_point_idx()[4:]
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[keypoints, 3] = 0
            pyflex.set_positions(pos.flatten())  # Nail one of the plates on the ground
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        angle = (np.random.random() - 0.5) * np.pi / 2
        self.rotate_particles(angle)
        if hasattr(self, 'action_tool'):
            x, y = np.mean(pyflex.get_positions().reshape((-1, 4))[self._get_key_point_idx()[:4]][:, (0, 2)], axis=0)
            x_off = np.random.random() * 0.1 - 0.05
            y_off = np.random.random() * 0.1 - 0.05
            self.action_tool.reset([x + x_off, 0.1, y + y_off])
            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)  # Per piece
        self.fold_group_a = np.array(list(range(num_particles)))
        self.fold_group_b = np.reshape(self.fold_group_a, [config['ClothSize'][0], config['ClothSize'][1]]) + num_particles
        # self.fold_group_b = np.flip(np.reshape(self.fold_group_a, [config['ClothSize'][0], config['ClothSize'][1]]), axis=0) + num_particles
        self.fold_group_b = np.reshape(self.fold_group_a, [config['ClothSize'][0], config['ClothSize'][1]]) + num_particles
        self.fold_group_b = self.fold_group_b.flatten()

        # Visualize Keypoints
        # self.set_test_color(len(pyflex.get_positions())// 4)
        # num_particles = len(pyflex.get_positions())// 4
        # colors = np.zeros(num_particles)
        # colors[self._get_key_point_idx()] = 5
        # self.set_colors(colors)
        # for i in range(100000):
        #     pyflex.step(render=True)

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def _step(self, action):
        # self.action_tool.visualize_picker_boundary()
        if self.action_mode == 'key_point':
            pyflex.step()
            action[2] = 0
            action[5] = 0
            action = np.array(action) / 10.
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            action = action.reshape([-1, 3])
            action = np.hstack([action, np.zeros([action.shape[0], 1])])
            cur_pos[self.action_key_point_idx, :] = last_pos[self.action_key_point_idx] + action
            pyflex.set_positions(cur_pos.flatten())
        else:
            self.action_tool.step(action)
            pyflex.step()

    def compute_reward(self, action=None, obs=None):
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
        curr_dist = np.minimum(np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)), 0.5) + \
                    1.2 * np.minimum(np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1)), 0.5)
        reward = -curr_dist
        return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        return {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
        }

    # def _set_to_folded(self):
    #     config = self.get_current_config()
    #     num_particles = np.prod(config['ClothSize'], dtype=int)
    #     particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here
    #
    #     cloth_dimx = config['ClothSize'][0]
    #     x_split = cloth_dimx // 2
    #     fold_group_a = particle_grid_idx[:, :x_split].flatten()
    #     fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()
    #
    #     curr_pos = pyflex.get_positions().reshape((-1, 4))
    #     curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
    #     curr_pos[fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.
    #
    #     pyflex.set_positions(curr_pos)
    #     for i in range(10):
    #         pyflex.step()
    #     return self._get_info()['performance']
