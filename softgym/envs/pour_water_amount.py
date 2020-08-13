import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import time
import copy
import os
from softgym.utils.misc import rotate_rigid_object, quatFromAxisAngle
from pyquaternion import Quaternion
import random
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml, pickle
import os.path as osp
import random


class PourWaterAmountPosControlEnv(PourWaterPosControlEnv):
    def __init__(self, observation_mode, action_mode, 
                config=None, cached_states_path='pour_water_amount_init_states.pkl', **kwargs):
        '''
        This class implements a pouring water task. Pouring a specific amount of water into the target cup.
        
        observation_mode: "cam_rgb" or "full_state"
        action_mode: "direct"
        
        '''

        super().__init__(observation_mode, action_mode, config, cached_states_path, **kwargs)

        # override the observation/state space to include the target amount
        if observation_mode in ['point_cloud', 'key_point']:
            if observation_mode == 'key_point':
                obs_dim = 12
                # z and theta of the second cup (poured_glass) does not change and thus are omitted.
                # Pos (x, y, z, theta) and shape (w, h, l) of the two cups, the water height and target amount
            else:
                max_particle_num = 12 * 12 * 48
                obs_dim = max_particle_num * 3
                self.particle_obs_dim = obs_dim
                obs_dim += 11 # Pos (x, y, z, theta) and shape (w, h, l) of the two cups, and target amount
            
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3 + 1),
                                         dtype=np.float32)

    def get_default_config(self):
        config = {
            'fluid': {
                'radius': 0.1,
                'rest_dis_coef': 0.55,
                'cohesion': 0.02,  # not actually used, instead, is computed as viscosity * 0.01
                'viscosity': 2.0,
                'surfaceTension': 0.,
                'adhesion': 0.0, # not actually used, instead, is computed as viscosity * 0.001
                'vorticityConfinement': 40,
                'solidpressure': 0.,
                'dim_x': 8,
                'dim_y': 18,
                'dim_z': 8,
            },
            'glass': {
                'border': 0.025,
                'height': 0.6,
                'glass_distance': 1.0,
                'poured_border': 0.025,
                'poured_height': 0.6,
            },
            "target_amount": 0.8,
        }
        return config

    def generate_env_variation(self, config, num_variations=5, save_to_file=False, **kwargs):
        """
        Just call PourWater1DPosControl's generate env variation, and then add the target amount.
        """

        super_config = copy.deepcopy(config)
        del super_config["target_amount"]
        cached_configs, cached_init_states = super().generate_env_variation(config=super_config, 
            num_variations=self.num_variations, save_to_file=False)
        
        for idx, cached_config in enumerate(cached_configs):
            cached_config['target_amount'] = 0.1 + np.random.uniform() * 0.9 # make sure pour at least 10% of water
            print("config {} target amount {}".format(idx, cached_config['target_amount']))

        self.cached_configs, self.cached_init_states = cached_configs, cached_init_states
        if save_to_file:
            combined = [cached_configs, cached_init_states]
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cached_configs, cached_init_states

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_rgb':
            ori_image = self.get_image(self.camera_width, self.camera_height)
            w, h, c = ori_image.shape
            new_image = np.zeros((w, h, c + 1))
            new_image[:, :, :c] = ori_image
            new_image[:, :, c].fill(self.current_config['target_amount'])

        elif self.observation_mode in ['point_cloud', 'key_point']:
            if self.observation_mode == 'point_cloud':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
                pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
                pos[:len(particle_pos)] = particle_pos
                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  self.current_config['targe_amount']])
            else:
                pos = np.empty(0, dtype=np.float)

                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  self._get_current_water_height(), self.current_config['target_amount']])
            
            return np.hstack([pos, cup_state]).flatten()
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None):
        """
        The reward is computed as the fraction of water in the poured glass.
        NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
        """
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        
        good_water = in_poured_glass * (1 - in_control_glass) # prevent to move the controlled cup directly into the target cup
        good_water_num = np.sum(good_water)
        target_water_num = int(water_num * self.current_config['target_amount'])
        diff = np.abs(target_water_num - good_water_num) / water_num

        reward = - diff
        return reward

    def _get_info(self):
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        
        good_water = in_poured_glass * (1 - in_control_glass) # prevent to move the controlled cup directly into the target cup
        good_water_num = np.sum(good_water)
        target_water_num = int(water_num * self.current_config['target_amount'])
        diff = np.abs(target_water_num - good_water_num) / water_num

        reward = - diff 

        return {
            'performance': reward,
            'target': self.current_config['target_amount']
        }