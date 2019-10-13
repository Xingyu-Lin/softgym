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

    def get_obs_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = 64 * 31
        return np.array([idx_p1, idx_p2])

    def get_action_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = 64 * 31
        return np.array([idx_p1, idx_p2])

    def set_scene(self):
        scene_params = np.array([])
        pyflex.set_scene(9, scene_params, 0)
        colors = self.get_colors()
        print(colors)
        # exit()
        colors[-len(colors)//2:] = 1
        self.set_colors(colors)

    def get_current_observation(self):
        pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        return pos[self.obs_key_point_idx, :3].flatten()

    def reset(self):
        self.set_state(self.init_state)

    def compute_reward(self, pos):
        return 0.

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
