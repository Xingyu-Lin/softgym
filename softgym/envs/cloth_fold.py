import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.flex_env import FlexEnv


class ClothFoldPointControlEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode):
        super().__init__()
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point']

        self.observation_mode = observation_mode
        self.action_mode = action_mode

        if observation_mode == 'key_point':
            self.observation_space = Box(np.array([-np.inf] * 6), np.array([np.inf] * 6), dtype=np.float32)
            self.obs_key_point_idx = self.get_obs_key_point_idx()
        else:
            raise NotImplementedError

        if action_mode == 'key_point':
            self.action_space = Box(np.array([-1.] * 6), np.array([1.] * 6), dtype=np.float32)
            self.action_key_point_idx = self.get_action_key_point_idx()
        else:
            raise NotImplementedError
        self.init_state = self.get_state()

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim

    def get_obs_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = self.cloth_xdim * (self.cloth_ydim - 1)
        return np.array([idx_p1, idx_p2])

    def get_action_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = self.cloth_xdim * (self.cloth_ydim - 1)
        return np.array([idx_p1, idx_p2])

    def set_scene(self):
        '''
        Setup the cloth scene and split particles into two groups for folding
        :return:
        '''
        self.cloth_xdim = 64
        self.cloth_ydim = 32

        scene_params = np.array([])
        pyflex.set_scene(9, scene_params, 0)

        # Set folding group
        particle_grid_idx = np.array(list(range(self.cloth_xdim * self.cloth_ydim))).reshape(self.cloth_ydim,
                                                                                             self.cloth_xdim)
        x_split = self.cloth_xdim // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = particle_grid_idx[:, self.cloth_xdim:x_split-1:-1].flatten()

        colors = np.zeros([self.cloth_ydim * self.cloth_xdim])
        colors[self.fold_group_b] = 1
        self.set_colors(colors)
        # self.set_test_color()

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

    def get_current_observation(self):
        pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        return pos[self.obs_key_point_idx, :3].flatten()

    def reset(self):
        self.set_state(self.init_state)

    def compute_reward(self, pos):
        '''
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        '''
        pos_group_a = pos[self.fold_group_a, :3]
        pos_group_b = pos[self.fold_group_b, :3]
        distance = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        return -distance

    def step(self, action):
        last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        pyflex.step()
        cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        action = action.reshape([-1, 3])
        action = np.hstack([action, np.zeros([action.shape[0], 1])])
        cur_pos[self.action_key_point_idx, :] = last_pos[self.action_key_point_idx] + action
        pyflex.set_positions(cur_pos.flatten())
        obs = self.get_current_observation()
        reward = self.compute_reward(cur_pos)
        return obs, reward, False, {}
