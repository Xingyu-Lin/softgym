import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env_new import RopeNewEnv
import scipy
import copy
from copy import deepcopy


class RopeFlattenNewEnv(RopeNewEnv):
    def __init__(self, cached_states_path='rope_flatten_new_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_endpoint_dist = None
        self.prev_distance_diff = None
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_states_path = osp.join(cur_dir, cached_states_path)
        else:
            self.cached_states_path = cached_states_path

        if not self.use_cached_states or self.get_cached_configs_and_states(cached_states_path) is False:
            config = self.get_default_config()
            self.generate_env_variation(config, self.num_variations, save_to_file=self.save_cache_states)
            success = self.get_cached_configs_and_states(cached_states_path)
            assert success

    def generate_env_variation(self, config=None, num_variations=1, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        default_config = config
        for i in range(num_variations):
            config = deepcopy(default_config)
            config['segment'] = self.get_random_rope_seg_num()
            self.set_scene(config)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            self._random_pick_and_place(pick_num=4, pick_scale=0.005)
            self._center_object()
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return generated_configs, generated_states

    def get_random_rope_seg_num(self):
        return np.random.randint(40, 41)

    def _reset(self):
        config = self.current_config
        # rope_init_pos = pyflex.get_positions().reshape(-1, 4)
        self.rope_length = config['segment'] * config['radius'] * 0.5
        # print("rope length is: ", self.rope_length)

        # set reward range
        self.reward_max = 0
        self.reward_min = -self.rope_length
        self.reward_range = self.reward_max - self.reward_min
        rope_particle_num = config['segment'] + 1
        self.key_point_indices = self._get_key_point_idx(rope_particle_num)

        self.prev_endpoint_dist = self._get_endpoint_distance()
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions().reshape([-1, 4])[4:] # a hack to remove the first 4 cloth particles
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.1, cy])
            # self.action_tool.init_particle_pos = rope_init_pos
            # self.action_tool.init_particle_pos = None

        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)[4:] # a hack to remove tha false cloth particles
        p1, p2 = pos[0, :3], pos[-1, :3]
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        curr_endpoint_dist = self._get_endpoint_distance()
        curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        if self.delta_reward:
            r = curr_distance_diff - self.prev_distance_diff
            if set_prev_reward:
                self.prev_distance_diff = curr_distance_diff
        else:
            r = curr_distance_diff
            # r = (r - self.reward_min) / self.reward_range # NOTE: this only suits to non-delta reward
        return r

    def _get_info(self):
        curr_endpoint_dist = self._get_endpoint_distance()
        curr_distance_diff = -np.abs(curr_endpoint_dist - self.rope_length)
        normalized_performance = (curr_distance_diff - self.reward_min) / self.reward_range
        return {
            'performance': curr_distance_diff,
            'normalized_performance': normalized_performance,
            'end_point_distance': curr_endpoint_dist
            }
