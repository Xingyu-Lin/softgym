import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from gym.spaces import Box
from softgym.envs.rope_flatten import RopeFlattenEnv
import scipy
import copy
from copy import deepcopy
import scipy.optimize as opt
import cv2

class RopeConfigurationEnv(RopeFlattenEnv):
    def __init__(self, cached_states_path='rope_configuration_init_states.pkl', reward_type='bigraph', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:

        manipulate the rope into a given character shape.
        """

        self.goal_characters = ['S', 'O', 'M', 'C', 'U']
        self.reward_type = reward_type
        super().__init__(cached_states_path=cached_states_path, **kwargs)
       
        # change observation space: add goal character information
        if self.observation_mode in ['key_point', 'point_cloud']:
            if self.observation_mode == 'key_point':
                obs_dim = 10 * 6 # evenly sample particles from current state and goal state
            else:
                max_particles = 160
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim * 2 # include both current particle position and goal particle position
            if self.action_mode in ['picker']:
                obs_dim += self.num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif self.observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3 * 2),
                                         dtype=np.float32) # stack current image and goal image
    def get_default_config(self, c='C'):
        """ Set the default config of the environment and load it to self.config """
        config = super().get_default_config()
        config['goal_character'] = c
        if c == 'M':
            config['segment'] = 60
        return config

    def _reset(self):
        super()._reset()

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        pyflex.step()
        return self._get_obs()

    def generate_env_variation(self, num_variations=1, **kwargs):
        """
        Just call RopeFlattenEnv's generate env variation, and then add the target character's position.
        """
        self.generate_alphabet_positions()
        self.generate_alphabet_image()
        super_config = copy.deepcopy(self.get_default_config())
        del super_config["goal_character"]
        cached_configs, cached_init_states = super().generate_env_variation(config=super_config, 
            num_variations=self.num_variations, save_to_file=False)
        
        for idx, cached_config in enumerate(cached_configs):
            goal_character = self.goal_characters[np.random.choice(len(self.goal_characters))]
            cached_config['goal_character'] = goal_character
            cached_config['goal_character_pos'] = self.goal_characters_position[goal_character]
            cached_config['goal_character_img'] = self.goal_characters_image[goal_character]
            print("config {} GoalCharacter {}".format(idx, goal_character))

        return cached_configs, cached_init_states
        

    def generate_alphabet_positions(self):
        self.goal_characters_position = {}
        cur_dir = osp.dirname(osp.abspath(__file__))
        character_loc_path = osp.join(cur_dir, '../files', 'rope_configuration.pkl')
        character_locs = pickle.load(open(character_loc_path, 'rb'))

        for c in character_locs:
            config = self.get_default_config(c=c)
            inv_mass = 1. / config['mass']
            radius = config['radius'] * config['scale']
            particle_num = int(config['segment'] + 1)

            pos = np.zeros((particle_num, 4))
            x, y = character_locs[c]
            if len(x) > particle_num:
                all_idxes = [x for x in range(1, len(x) - 1)]
                chosen_idxes = np.random.choice(all_idxes, particle_num - 2, replace=False)
                chosen_idxes = list(np.sort(chosen_idxes))
                chosen_idxes = [0] + chosen_idxes + [len(x) - 1]
                x = np.array(x)[chosen_idxes]
                y = np.array(y)[chosen_idxes]
            elif particle_num > len(x):
                interpolate_idx = np.random.choice(range(1, len(x) - 1), particle_num - len(x), replace=False)
                interpolate_idx = list(np.sort(interpolate_idx))
                interpolate_idx = [0] + interpolate_idx
                x_new = []
                y_new = []
                print('interpolate_idx: ', interpolate_idx)
                for idx in range(1, len(interpolate_idx)):
                    [x_new.append(x[_]) for _ in range(interpolate_idx[idx - 1], interpolate_idx[idx])]
                    [y_new.append(y[_]) for _ in range(interpolate_idx[idx - 1], interpolate_idx[idx])]
                    print(interpolate_idx[idx])
                    [print(_, end=' ') for _ in range(interpolate_idx[idx - 1], interpolate_idx[idx])]
                    x_new.append((x[interpolate_idx[idx]] + x[interpolate_idx[idx] + 1]) / 2)
                    y_new.append((y[interpolate_idx[idx]] + y[interpolate_idx[idx] + 1]) / 2)
                [x_new.append(x[_]) for _ in range(interpolate_idx[-1], len(x))]
                [y_new.append(y[_]) for _ in range(interpolate_idx[-1], len(y))]
                [print(_, end=' ') for _ in range(interpolate_idx[-1], len(y))]
                x = x_new
                y = y_new

            for p_idx in range(particle_num):
                pos[p_idx][0] = y[p_idx] * radius
                pos[p_idx][1] = 0.05 # y
                pos[p_idx][2] = x[p_idx] * radius
                pos[p_idx][3] = inv_mass

            pos[:, 0] -= np.mean(pos[:, 0])
            pos[:, 2] -= np.mean(pos[:, 2])

            self.goal_characters_position[c] = pos.copy()

    def generate_alphabet_image(self):
        self.goal_characters_image = {}
        for c in self.goal_characters:
            default_config = self.get_default_config(c=c)
            goal_c_pos =  self.goal_characters_position[c]
            self.set_scene(default_config)
            all_positions = pyflex.get_positions().reshape([-1, 4])
            all_positions = goal_c_pos.copy() # ignore the first a few cloth particles
            pyflex.set_positions(all_positions)
            self.update_camera('default_camera', default_config['camera_params']['default_camera']) # why we need to do this?
            self.action_tool.reset([0., -1., 0.]) # hide picker
            # goal_c_img = self.get_image(self.camera_height, self.camera_width)

            # import time
            # for _ in range(50):   
            #     pyflex.step(render=True)
                # time.sleep(0.1)
                # cv2.imshow('img', self.get_image())
                # cv2.waitKey()
                
            goal_c_img = self.get_image(self.camera_height, self.camera_width)
            # cv2.imwrite('../data/images/rope-configuration-goal-image-{}.png'.format(c), goal_c_img[:,:,::-1])
            # exit()

            self.goal_characters_image[c] = goal_c_img.copy()

    def compute_reward(self, action=None, obs=None, **kwargs):
        """ Reward is the matching degree to the goal character"""
        goal_c_pos = self.current_config["goal_character_pos"][:, :3]
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
        # way1: index matching
        if self.reward_type == 'index':
            dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
            reward = -np.mean(dist)

        if self.reward_type == 'bigraph':
            # way2: downsample and then use Hungarian algorithm for bipartite graph  matching
            downsampled_cur_pos = current_pos[self.key_point_indices]
            downsampled_goal_pos = goal_c_pos[self.key_point_indices]
            W = np.zeros((len(downsampled_cur_pos), len(downsampled_cur_pos)))
            for idx in range(len(downsampled_cur_pos)):
                all_dist = np.linalg.norm(downsampled_cur_pos[idx] - downsampled_goal_pos, axis=1)
                W[idx, :] = all_dist
            
            row_idx, col_idx = opt.linear_sum_assignment(W)
            dist = W[row_idx, col_idx].sum()
            reward = -dist / len(downsampled_goal_pos)
        
        return reward
            

    def _get_obs(self):
        if self.observation_mode == 'cam_rgb':
            obs_img = self.get_image(self.camera_height, self.camera_width)
            goal_img = self.current_config['goal_character_img']
            ret_img = np.concatenate([obs_img, goal_img], axis=2)
            return ret_img

        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
            pos[len(particle_pos):] = self.current_config["goal_character_pos"][:, :3].flatten()
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self.key_point_indices, :3]
            goal_keypoint_pos = self.current_config["goal_character_pos"][self.key_point_indices, :3]
            pos = np.concatenate([keypoint_pos, goal_keypoint_pos], axis=0).flatten()

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, :3].flatten()])
        return pos

    def _get_info(self):
        goal_c_pos = self.current_config["goal_character_pos"][:, :3]
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
        # way1: index matching
        if self.reward_type == 'index':
            dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
            reward = -np.mean(dist)

        if self.reward_type == 'bigraph':
            # way2: downsample and then use Hungarian algorithm for bipartite graph  matching
            downsampled_cur_pos = current_pos[self.key_point_indices]
            downsampled_goal_pos = goal_c_pos[self.key_point_indices]
            W = np.zeros((len(downsampled_cur_pos), len(downsampled_cur_pos)))
            for idx in range(len(downsampled_cur_pos)):
                all_dist = np.linalg.norm(downsampled_cur_pos[idx] - downsampled_goal_pos, axis=1)
                W[idx, :] = all_dist
            
            row_idx, col_idx = opt.linear_sum_assignment(W)
            dist = W[row_idx, col_idx].sum()
            reward = -dist / len(downsampled_goal_pos)

        performance = reward
        performance_init =  performance if self.performance_init is None else self.performance_init  # Use the original performance

        return {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (self.reward_max - performance_init),
        }
