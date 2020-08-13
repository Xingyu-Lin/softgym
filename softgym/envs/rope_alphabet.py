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

class RopeAlphaBetEnv(RopeFlattenEnv):
    def __init__(self, cached_states_path='rope_alphabet_init_states.pkl', reward_type='bigraph', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:

        manipulate the rope into a given character shape.
        """

        self.goal_characters = ['C']
        self.reward_type = reward_type
        super().__init__(cached_states_path=cached_states_path, **kwargs)
       
        # change observation space: add goal character information
        if self.observation_mode in ['key_point', 'point_cloud']:
            if self.observation_mode == 'key_point':
                obs_dim = len(self._get_key_point_idx()) * 6 # evenly sample particles from current state and goal state
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
    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClusterSpacing': 1.5,
            'ClusterRadius': 0.,
            'ClusterStiffness': 0.55,
            'DynamicFriction': 3.0,
            'ParticleFriction': 0.25,
            'ParticleInvMass': 0.01,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': np.array([0., 7., 3.]),
                                   'angle': np.array([0, -65 / 180. * np.pi, 0.]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'GoalCharacter': 'O'
        }
        return config


    def generate_env_variation(self, config=None, num_variations=1, save_to_file=False, **kwargs):
        """
        Just call RopeFlattenEnv's generate env variation, and then add the target character's position.
        """
        self.generate_alphabet_positions()
        self.generate_alphabet_image()
        super_config = copy.deepcopy(config)
        del super_config["GoalCharacter"]
        cached_configs, cached_init_states = super().generate_env_variation(config=super_config, 
            num_variations=self.num_variations, save_to_file=False)
        
        for idx, cached_config in enumerate(cached_configs):
            goal_character = self.goal_characters[np.random.choice(len(self.goal_characters))]
            cached_config['GoalCharacter'] = goal_character
            cached_config['GoalCharacterPos'] = self.goal_characters_position[goal_character]
            cached_config['GoalCharacterImg'] = self.goal_characters_image[goal_character]
            print("config {} GoalCharacter {}".format(idx, goal_character))

        self.cached_configs, self.cached_init_states = cached_configs, cached_init_states
        if save_to_file:
            combined = [self.cached_configs, self.cached_init_states]
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cached_configs, cached_init_states
        

    def generate_alphabet_positions(self, rope_len=4.88, particle_num=160):
        inv_mass = self.get_default_config()['ParticleInvMass']
        self.goal_characters_position = {}
        for c in self.goal_characters:
            pos = np.zeros((particle_num, 4))
            if c == 'O':
                r = rope_len / (2 * np.pi)
                radius_unit = 2 * np.pi / particle_num
                for p_idx in range(particle_num):
                    pos[p_idx][0] = r * np.cos(radius_unit * p_idx) # x
                    pos[p_idx][1] = 0.05 # y
                    pos[p_idx][2] = -r * np.sin(radius_unit * p_idx) # z. 
                    pos[p_idx][3] = inv_mass
            
            elif c == 'C':
                r = rope_len / (2 * np.pi) * 1.5
                radius_unit = 2 * 2 / 3 * np.pi / particle_num
                base = np.pi / 3
                for p_idx in range(particle_num):
                    pos[p_idx][0] = r * np.cos(base + radius_unit * p_idx) # x
                    pos[p_idx][1] = 0.05 # y
                    pos[p_idx][2] = -r * np.sin(base + radius_unit * p_idx) # z. 
                    pos[p_idx][3] = inv_mass

            self.goal_characters_position[c] = pos.copy()

    def generate_alphabet_image(self):
        self.goal_characters_image = {}
        default_config = self.get_default_config()
        for c in self.goal_characters:
            goal_c_pos =  self.goal_characters_position[c]
            self.set_scene(default_config)
            pyflex.set_positions(goal_c_pos)
            self.update_camera('default_camera', default_config['camera_params']['default_camera']) # why we need to do this?
            self.action_tool.reset([0., -1., 0.]) # hide picker
            goal_c_img = self.get_image(self.camera_height, self.camera_width)
            self.goal_characters_image[c] = goal_c_img.copy()

    def compute_reward(self, action=None, obs=None):
        """ Reward is the matching degree to the goal character"""
        goal_c_pos = self.current_config["GoalCharacterPos"][:, :3]
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
        # way1: index matching
        if self.reward_type == 'index':
            dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
            reward = -np.mean(dist)

        if self.reward_type == 'bigraph':
            # way2: downsample and then use Hungarian algorithm for bipartite graph  matching
            downsampled_cur_pos = current_pos[self._get_key_point_idx()]
            downsampled_goal_pos = goal_c_pos[self._get_key_point_idx()]
            W = np.zeros((len(downsampled_cur_pos), len(downsampled_cur_pos)))
            for idx in range(len(downsampled_cur_pos)):
                all_dist = np.linalg.norm(downsampled_cur_pos[idx] - downsampled_goal_pos, axis=1)
                W[idx, :] = all_dist
            
            row_idx, col_idx = opt.linear_sum_assignment(W)
            dist = W[row_idx, col_idx].sum()
            reward = -dist / len(downsampled_goal_pos)
        
        # print("index matching reward: ", reward1)
        # print("bigraph matching reward: ", reward2)
        return reward
            

    def _get_obs(self):
        goal_c = self.current_config["GoalCharacter"]
        if self.observation_mode == 'cam_rgb':
            obs_img = self.get_image(self.camera_height, self.camera_width)
            goal_img = self.current_config['GoalCharacterImg']
            ret_img = np.concatenate([obs_img, goal_img], axis=2)
            return ret_img

        if self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
            pos[len(particle_pos):] = self.current_config["GoalCharacterPos"][:, :3]
        elif self.observation_mode == 'key_point':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            goal_keypoint_pos = self.current_config["GoalCharacterPos"][self._get_key_point_idx(), :3]
            pos = np.concatenate([keypoint_pos, goal_keypoint_pos], axis=0)

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, :3].flatten()])
        return pos

    def _get_info(self):
        goal_c_pos = self.current_config["GoalCharacterPos"][:, :3]
        current_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        
        if self.reward_type == 'index':
            dist = np.linalg.norm(current_pos - goal_c_pos, axis=1)
            reward = -np.mean(dist)

        if self.reward_type == 'bigraph':
            # way2: downsample and then use Hungarian algorithm for bipartite graph  matching
            downsampled_cur_pos = current_pos[self._get_key_point_idx()]
            downsampled_goal_pos = goal_c_pos[self._get_key_point_idx()]
            W = np.zeros((len(downsampled_cur_pos), len(downsampled_cur_pos)))
            for idx in range(len(downsampled_cur_pos)):
                all_dist = np.linalg.norm(downsampled_cur_pos[idx] - downsampled_goal_pos, axis=1)
                W[idx, :] = all_dist
            
            row_idx, col_idx = opt.linear_sum_assignment(W)
            dist = W[row_idx, col_idx].sum()
            reward = -dist / len(downsampled_goal_pos)

        return {'performance': reward}
