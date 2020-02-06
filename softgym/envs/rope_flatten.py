import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.rope_env import RopeEnv
import scipy
import copy
from copy import deepcopy


class RopeFlattenEnv(RopeEnv):
    def __init__(self, cached_states_path='rope_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_endpoint_dist = None
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

    def generate_env_variation(self, num_variations=1, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        for i in range(num_variations):
            config = deepcopy(default_config)
            self.set_scene(config)

            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, 3] = config['ParticleInvMass']
            pyflex.set_positions(pos)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            self._random_pick_and_place(pick_num=10)
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
        self.prev_endpoint_dist = self._get_endpoint_distance()
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.2, cy])
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('picker'):
            self.action_tool.step(action)
            pyflex.step()
        else:
            raise NotImplementedError
        return

    def _get_endpoint_distance(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        p1, p2 = pos[0, :3], pos[-1, :3]
        # print('end point distance:', p1, ' ', p2, ' ', np.linalg.norm(p1 - p2).squeeze())
        return np.linalg.norm(p1 - p2).squeeze()

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ Reward is the distance between the endpoints of the rope"""
        curr_endpoint_dist = self._get_endpoint_distance()
        if self.delta_reward:
            r = curr_endpoint_dist - self.prev_endpoint_dist
            if set_prev_reward:
                self.prev_endpoint_dist = curr_endpoint_dist
        else:
            r = curr_endpoint_dist
        return r

    def _get_info(self):
        curr_endpoint_dist = self._get_endpoint_distance()
        return {'performance': curr_endpoint_dist}
