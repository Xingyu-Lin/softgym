import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv
import copy
from copy import deepcopy


class ClothDropEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_drop_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        assert self.action_tool.num_picker == 2  # Two drop points for this task
        self.prev_dist = None  # Should not be used until initialized
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_states_path = osp.join(cur_dir, cached_states_path)
        else:
            self.cached_states_path = cached_states_path
        success = self.get_cached_configs_and_states(cached_states_path)
        if not success or not self.use_cached_states:
            self.generate_env_variation(self.num_variations, save_to_file=True)
            success = self.get_cached_configs_and_states(cached_states_path)
            assert success

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
            # 'camera_params': {'default_camera':
            #                       {'pos': np.array([0, 7, 0.]),
            #                        'angle': np.array([0., -90. / 180. * np.pi, 0.]),
            #                        'width': self.camera_width,
            #                        'height': self.camera_height}}
        }
        return config

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def generate_env_variation(self, num_variations=1, save_to_file=False, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
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
            for j in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j!=0:
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
        self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos)
            picker_low = middle_point - [0.2, 0.1, 0.5]
            picker_high = middle_point + [0.5, 0.1, 0.5]
            self.action_tool.update_picker_boundary(picker_low, picker_high)
        return self._get_obs()

    def _step(self, action):
        # self.action_tool.visualize_picker_boundary()
        # while (1):
        #     pyflex.step()
        if self.action_mode.startswith('key_point'):
            # TODO ad action_repeat
            print('Need to add action repeat')
            raise NotImplementedError
            raise DeprecationWarning
            valid_idxs = np.array([0, 63, 31 * 64, 32 * 64 - 1])
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            pyflex.step()

            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            action = action.reshape([-1, 4])
            idxs = np.hstack(action[:, 0])
            updates = action[:, 1:]
            action = np.hstack([action, np.zeros([action.shape[0], 1])])
            vels = pyflex.get_velocities()
            cur_pos[:, 3] = 1
            if self.action_mode == 'key_point_pos':
                cur_pos[valid_idxs[idxs.astype(int)], :3] = last_pos[valid_idxs[idxs.astype(int)]][:, :3] + updates
                cur_pos[valid_idxs[idxs.astype(int)], 3] = 0


            else:
                vels = np.array(vels).reshape([-1, 3])
                vels[idxs.astype(int), :] = updates
            pyflex.set_positions(cur_pos.flatten())
            pyflex.set_velocities(vels.flatten())
        else:
            self.action_tool.step(action)
            pyflex.step()
        return

    def _get_current_dist(self, pos):
        target_pos = self.get_current_config()['target_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - target_pos, axis=1))
        return curr_dist

    # def _get_center_point(self, pos):
    #     pos = np.reshape(pos, [-1, 4])
    #     min_x = np.min(pos[:, 0])
    #     min_y = np.min(pos[:, 2])
    #     max_x = np.max(pos[:, 0])
    #     max_y = np.max(pos[:, 2])
    #     return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        if self.delta_reward:
            r = self.prev_dist - curr_dist
            if set_prev_reward:
                self.prev_dist = curr_dist
        else:
            r = - curr_dist
        return r

    @property
    def performance_bound(self):
        max_dist = 1.043
        min_p = - max_dist
        max_p = 0
        return min_p, max_p

    def _get_info(self):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        performance = -curr_dist
        pb = self.performance_bound
        return {'performance': performance,
                'normalized_performance': (performance - pb[0]) / (pb[1] - pb[0])}
