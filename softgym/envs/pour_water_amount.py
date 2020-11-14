import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.pour_water import PourWaterPosControlEnv
import copy
from softgym.utils.misc import quatFromAxisAngle
import pickle


class PourWaterAmountPosControlEnv(PourWaterPosControlEnv):
    def __init__(self, observation_mode, action_mode, 
                config=None, cached_states_path='pour_water_amount_init_states.pkl', **kwargs):
        '''
        This class implements a pouring water task. Pouring a specific amount of water into the target cup.
        
        observation_mode: "cam_rgb" or "point_cloud"
        action_mode: "direct"
        
        '''

        super().__init__(observation_mode, action_mode, config, cached_states_path, **kwargs)

        # override the observation/state space to include the target amount
        if observation_mode in ['point_cloud', 'key_point']:
            if observation_mode == 'key_point':
                obs_dim = 13
                # z and theta of the second cup (poured_glass) does not change and thus are omitted.
                # Pos (x, y, z, theta) and shape (w, h, l) of the two cups, the water height and target amount
            else:
                max_particle_num = 13 * 13 * 13 * 3
                obs_dim = max_particle_num * 3
                self.particle_obs_dim = obs_dim
                obs_dim += 12 # Pos (x, y, z, theta) and shape (w, h, l) of the two cups, and target amount
            
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3 + 1),
                                         dtype=np.float32)

    def get_default_config(self):
        config = super().get_default_config()
        config['target_height'] = 0.8
        return config

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'glass_x': self.glass_x, 'glass_y': self.glass_y, 'glass_rotation': self.glass_rotation,
                'glass_states': self.glass_states, 'poured_glass_states': self.poured_glass_states,
                'glass_params': self.glass_params, 'config_id': self.current_config_id, 
                'line_box_x': self.line_box_x, 'line_box_y': self.line_box_y}

    def set_state(self, state_dic):
        self.line_box_x = state_dic['line_box_x']
        self.line_box_y = state_dic['line_box_y']
        super().set_state(state_dic)

    def set_shape_states(self, glass_states, poured_glass_states):
        all_states = np.concatenate((glass_states, poured_glass_states), axis=0)

        if self.line_box_x is not None:
            quat = quatFromAxisAngle([0, 0, -1.], 0.)
            indicator_box_line_states = np.zeros((1, self.dim_shape_state))

            indicator_box_line_states[0, :3] = np.array([self.line_box_x, self.line_box_y, 0.])
            indicator_box_line_states[0, 3:6] = np.array([self.line_box_x, self.line_box_y, 0.])
            indicator_box_line_states[:, 6:10] = quat
            indicator_box_line_states[:, 10:] = quat

            all_states = np.concatenate((all_states, indicator_box_line_states), axis=0)
        
        pyflex.set_shape_states(all_states)
        if self.line_box_x is not None:
            pyflex.step(render=True)
            # time.sleep(20)

    def set_scene(self, config, states=None):
        self.line_box_x = self.line_box_y = None

        if states is None:
            super().set_scene(config=config, states=states, create_only=False) # this adds the water, the controlled cup, and the target cup.
        else:
            super().set_scene(config=config, states=states, create_only=True) # this adds the water, the controlled cup, and the target cup.

        # needs to add an indicator box for how much water we want to pour into the target cup
        # create an line on the left wall of the target cup to indicate how much water we want to pour into it.
        halfEdge = np.array([0.005 / 2., 0.005 / 2., (self.poured_glass_dis_z - 2 * self.poured_border) / 2.])
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        # print("pourwater amount add trigger box")
        pyflex.add_box(halfEdge, center, quat, 1) # set trigger to be true to create a different color for the indicator box.

        max_water_height = self._get_current_water_height() - self.border / 2
        controlled_size = (self.glass_dis_x) * (self.glass_dis_z)
        target_size = (self.poured_glass_dis_x) * (self.poured_glass_dis_z)
        estimated_target_water_height = max_water_height * config['target_amount'] / (target_size / controlled_size)
        self.line_box_x = self.x_center + self.glass_distance - self.poured_glass_dis_x / 2
        self.line_box_y = self.poured_border * 0.5 + estimated_target_water_height


        if states is None:
            self.set_shape_states(self.glass_states, self.poured_glass_states)
        else:
            self.set_state(states)

    def generate_env_variation(self, num_variations=5, **kwargs):
        """
        Just call PourWater1DPosControl's generate env variation, and then add the target amount.
        """
        config = self.get_default_config()
        super_config = copy.deepcopy(config)
        super_config['target_amount'] = np.random.uniform(0.2, 1)
        cached_configs, cached_init_states = super().generate_env_variation(config=super_config, num_variations=self.num_variations)
        return cached_configs, cached_init_states

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_width, self.camera_height)

        elif self.observation_mode in ['point_cloud', 'key_point']:
            if self.observation_mode == 'point_cloud':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
                pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
                pos[:len(particle_pos)] = particle_pos
                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  self.line_box_y, self.current_config['targe_amount']])
            else:
                pos = np.empty(0, dtype=np.float)

                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  self._get_current_water_height(), self.line_box_y, self.current_config['target_amount']])
            
            return np.hstack([pos, cup_state]).flatten()
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None, **kwargs):
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
        # Duplicate of the compute reward function!
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        good_water = in_poured_glass * (1 - in_control_glass)
        good_water_num = np.sum(good_water)
        target_water_num = int(water_num * self.current_config['target_amount'])
        diff = np.abs(target_water_num - good_water_num) / water_num

        reward = - diff 

        performance = reward
        performance_init =  performance if self.performance_init is None else self.performance_init  # Use the original performance

        return {
            'normalized_performance': (performance - performance_init) / (self.reward_max - performance_init),
            'performance': performance,
            'target': self.current_config['target_amount']
        }