import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv


class ClothFoldEnv(ClothEnv):
    def __init__(self, cached_init_state_path='cloth_fold_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(config_file="ClothFoldConfig.yaml", **kwargs)
        self.cached_init_state = []

        if cached_init_state_path.startswith('/'):
            self.cached_init_state_path = cached_init_state_path
        else:
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_init_state_path = osp.join(cur_dir, cached_init_state_path)

        if osp.exists(self.cached_init_state_path):
            self._load_init_state()
            print('ClothFoldEnv: {} cached initial states loaded'.format(len(cached_init_state_path)))



    def initialize_camera(self):
        '''
        set the camera width, height, ition and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        self.camera_params = {
            'pos': np.array([0., 3, 3.5]),
            'angle': np.array([0, -45 / 180. * np.pi, 0.]),
            'width': self.camera_width,
            'height': self.camera_height
        }

    def _load_init_state(self, init_state_path):
        cur_dir = osp.dirname(osp.abspath(__file__))
        with open(osp.join(cur_dir, init_state_path), "rb") as handle:
            self.cached_init_state = pickle.load(handle)

    def generate_init_state(self, num_init_state=1, save_to_file=False):
        """ Generate initial states. Note: This will also change the current states! """
        # TODO Xingyu: Add options for generating initial states with different parameters.
        # TODO additionally, can vary the height / number of pick point
        original_state = self.get_state()
        num_particle = original_state['particle_pos'].reshape((-1, 4)).shape[0]
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.03  # Cloth stable when all particles' vel are smaller than this
        init_states = []

        for i in range(num_init_state):
            # Drop the cloth and wait to stablize
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break

            if self.action_mode == 'sphere' or self.action_mode == 'picker':
                curr_pos = pyflex.get_positions()
                center_point = num_particle // 2
                self.action_tool.reset(curr_pos[center_point * 4:center_point * 4 + 3] + [0., 0.2, 0.])

            init_states.append(self.get_state())
            self.set_state(original_state)

        if save_to_file:
            with open(self.cached_init_state_path, 'wb') as handle:
                pickle.dump(init_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return init_states

    def set_scene(self):
        """ Setup the cloth scene and split particles into two groups for folding """
        super().set_scene()
        # Set folding group
        particle_grid_idx = np.array(list(range(self.cloth_xdim * self.cloth_ydim))).reshape(self.cloth_ydim,
                                                                                             self.cloth_xdim)

        x_split = self.cloth_xdim // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = particle_grid_idx[:, self.cloth_xdim:x_split - 1:-1].flatten()

        colors = np.zeros([self.cloth_ydim * self.cloth_xdim])
        colors[self.fold_group_b] = 1

        self.set_colors(colors)
        # self.set_test_color()
        # print("scene set")

    def set_test_color(self):
        '''
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        '''
        colors = np.zeros((self.cloth_xdim * self.cloth_ydim))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def _reset(self):
        """ Right now only use one initial state"""
        if len(self.cached_init_state) == 0:
            state_dicts = self.generate_init_state(1)
            self.cached_init_state.extend(state_dicts)
        cached_id = np.random.randint(len(self.cached_init_state))
        self.set_state(self.cached_init_state[cached_id])

        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0, 1, 0])
        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        return self._get_obs()

    def compute_reward(self, pos, set_prev_dist=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b_init, axis=1))
        reward = self.prev_dist - curr_dist
        if set_prev_dist:
            self.prev_dist = curr_dist
        return reward

    def _step(self, action):
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
            for _ in range(self.action_repeat):
                pyflex.step()
                self.action_tool.step(action)
        pos = pyflex.get_positions()
        reward = self.compute_reward(pos, set_prev_dist=True)
        obs = self._get_obs()
        return obs, reward, False, {}
