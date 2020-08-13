import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv


class RopeClothStraightenEnv(ClothEnv):
    def __init__(self, cached_states_path='ropeCloth_straighten_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        super().__init__(**kwargs)

        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_states_path = osp.join(cur_dir, cached_states_path)
        else:
            self.cached_states_path = cached_states_path

        if self.use_cached_states:
            success = self.get_cached_configs_and_states(cached_states_path)

        if not self.use_cached_states or not success:
            self.cached_configs, self.cached_init_states = self.generate_env_variation(self.num_variations, save_to_file=self.save_cache_states)
            success = self.get_cached_configs_and_states(cached_states_path)
            assert success

    def _sample_cloth_size(self):
        return np.random.randint(60, 120), 2

    def generate_env_variation(self, num_variations=2, save_to_file=False, vary_cloth_size=True, config=None):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        if config is None:
            default_config = self.get_default_config()
        else:
            default_config = config
        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            print("generating configuration {}, dimx is {}, dimy is {}".format(i, cloth_dimx, cloth_dimy))
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()
                pyflex.render()

            self._random_pick_and_place(pick_num=5, pick_scale=0.005)
            self._center_object()

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return generated_configs, generated_states

    def _get_key_point_idx(self, num):
        if self.key_point_indices is not None:
            return self.key_point_indices
        cur_idx = 0
        indices = []
        while cur_idx < num:
            indices.append(cur_idx)
            cur_idx += 10
        if num - 1 not in indices:
            indices.append(num - 1)
        return indices

    def _reset(self):
        pos = pyflex.get_positions().reshape((-1, 4))  # x coordinate of left-top corner
        if hasattr(self, 'action_tool'):
            x = pos[0][0]
            self.action_tool.reset([x, 0.1, 0])
    
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        self.initial_rope_len = num_particles * self.cloth_particle_radius
        self.key_point_indices = None
        self.key_point_indices = self._get_key_point_idx(num_particles)
        self.prev_diff = np.abs(np.linalg.norm(pos[0] - pos[-1]) - self.initial_rope_len)

        self.reward_min = - self.initial_rope_len
        self.reward_max = 0
        return self._get_obs()

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

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """
        reward is the lenght difference to the initial rope length
        """
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        
        curr_len = np.linalg.norm(pos[0] - pos[-1])
        len_difference = np.abs(curr_len - self.initial_rope_len)

        if self.delta_reward:
            reward = self.prev_diff - len_difference
            if set_prev_reward:
                self.prev_diff = len_difference
        else:
            reward = - len_difference
        return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        
        curr_len = np.linalg.norm(pos[0] - pos[-1])
        len_difference = np.abs(curr_len - self.initial_rope_len)
        performance = - len_difference


        return {
            'performance': performance,
            'normalized_performance': (performance - self.reward_min) / self.reward_max,
            'current len': curr_len
        }
