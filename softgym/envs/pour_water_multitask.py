import numpy as np
from gym.spaces import Box, Dict

import pyflex
from softgym.envs.fluid_env import FluidEnv
from softgym.envs.pour_water import PourWaterPosControlEnv
import time
import copy
import os
from softgym.envs.util import rotate_rigid_object
from pyquaternion import Quaternion
import random
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os.path as osp
from softgym.core.multitask_env import MultitaskEnv

class PourWaterPosControlGoalConditionedEnv(PourWaterPosControlEnv, MultitaskEnv):
    def __init__(self, observation_mode = 'full_state', action_mode = 'direct', **kwargs):
        '''
        This class implements a pouring water task.
        
        observation_mode: "cam_img" or "full_state"
        action_mode: "direct"
        horizon: environment horizon
        
        TODO: add more description of the task.
        TODO: allow parameter configuring of the scence.
        '''

        self.state_dict_goal = None

        assert observation_mode in ['full_state'] 
        assert action_mode in ['direct'] 
        PourWaterPosControlEnv.__init__(self, observation_mode = observation_mode, action_mode = action_mode, **kwargs)

        self.fluid_num = pyflex.get_n_particles()
        self.obs_box = Box(low = -np.inf, high = np.inf, shape = ((self.dim_position + self.dim_velocity) * self.fluid_num + 7 * 2, ), 
            dtype=np.float32) # the last 12 dim: 6 for each cup, x, y, z, theta, width, length, height
        self.goal_box = Box(low = -2, high = 2, shape = ((self.dim_position + self.dim_velocity) * self.fluid_num + 7 * 2, ), 
            dtype=np.float32) # make water particles being in some range

        self.observation_space = Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),
            ('achieved_goal', self.goal_box),
            ('state_achieved_goal', self.goal_box),
        ])

    def sample_goals(self, batch_size):
        '''
        currently there is only one goal that can be sampled, where all the water particles are directly set to be in the
        target cup.
        '''
        assert batch_size == 1, "for a fixed task configuration, we can only sample 1 target goal!"

        # make the wate being like a cubic
        fluid_pos = np.ones((self.fluid_num, self.dim_position))
        fluid_vel = np.zeros((self.fluid_num, self.dim_velocity))
        
        # lower = np.random.uniform(self.goal_box.low, self.goal_box.high, size = (1, self.dim_position))
        
        # goal: water all inside target cup
        lower_x = self.glass_params['poured_glass_x_center'] - self.glass_params['poured_glass_dis_x'] / 3.
        lower_z = -self.glass_params['poured_glass_dis_z'] / 3
        lower_y = self.glass_params['poured_border'] 
        lower = np.array([lower_x, lower_y, lower_z])
        cnt = 0
        for x in range(self.fluid_params['dim_x']):
            for y in range(self.fluid_params['dim_y']):
                for z in range(self.fluid_params['dim_z']):
                    fluid_pos[cnt][:3] = lower + np.array([x, y, z])*self.fluid_params['radius'] / 2#+ np.random.rand() * 0.01
                    cnt += 1

        fluid_goal = np.concatenate((fluid_pos.reshape(1, -1), fluid_vel.reshape(1, -1)), axis = 1)

        # the full goal state includes the control cup and target cup's state
        # make control cup hang near the target cup and rotates towards the target cup, simulating a real pouring.
        pouring_glass_x = lower_x - self.glass_params['poured_height']
        pouring_glass_z = 0.
        pouring_glass_y = 2.2 * self.glass_params['poured_height']
        pouring_theta = 0.6 * np.pi

        pouring_glass_goal = np.array([pouring_glass_x, pouring_glass_y, pouring_glass_z, pouring_theta, self.glass_params['glass_dis_x'], 
            self.glass_params['glass_dis_z'], self.glass_params['height']])
        poured_glass_goal = self.get_poured_glass_state()
        glass_goal = np.concatenate((pouring_glass_goal.reshape(1,-1), poured_glass_goal.reshape(1,-1)), axis = 1)

        goal = np.concatenate((fluid_goal, glass_goal), axis = 1)

        self.state_goal = goal

        return {
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def get_poured_glass_state(self):
        return np.array([self.glass_params['poured_glass_x_center'], self.glass_params['poured_border'], 0, 0,
            self.glass_params['poured_glass_dis_x'], self.glass_params['poured_glass_dis_z'], self.glass_params['poured_height']])


    def compute_reward(self, action, obs, info = None):
        '''
        reward is the l2 distance between the goal state and the current state.
        '''
        r = -np.linalg.norm(
            obs['state_achieved_goal'] - obs['state_desired_goal'])
        return r

    def compute_rewards(self, action, obs, info = None):
        '''
        rewards in state space.
        '''
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        return -dist   
        
    def _reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        # NOTE: only suits for skewfit algorithm, because we are not actually sampling from this
        # true underlying env, but only sample from the vae latents. This reduces overhead to sample a goal each time for now.
        if self.state_dict_goal is None:
            self.state_dict_goal = self.sample_goal() 

        PourWaterPosControlEnv.reset(self)       
        return self._get_obs()
    
    def get_env_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        a wrapper to be compatiable with the MultiTask Env.
        '''
        return self.get_state()

    def set_env_state(self, state_dic): 
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        a wrapper to be compatiable with the MultiTask Env.
        '''
        return self.set_state(state_dic)
    
    def set_to_goal(self, goal):
        '''
        given a goal, set the flex state to be that goal.
        needed by image env to sample goals.
        '''
        state_goal = goal['state_desired_goal']
        particle_pos = state_goal[:self.fluid_num * self.dim_position]
        particle_vel = state_goal[self.fluid_num * self.dim_position: (self.dim_position + self.dim_velocity) * self.fluid_num]
        
        tmp = (self.dim_position + self.dim_velocity) * self.fluid_num
        control_cup_x, control_cup_y, control_cup_z, control_cup_theta = \
                state_goal[tmp], state_goal[tmp + 1], state_goal[tmp+2], state_goal[tmp+3]     
     
        # move controled cup tp target position
        tmp_states = self.glass_states
        diff_x = control_cup_x - self.glass_x
        diff_y = control_cup_y - self.glass_y
        diff_theta = control_cup_theta - self.glass_rotation
        steps = 200        
        for i in range(steps):
            glass_new_states = self.rotate_glass(tmp_states, diff_x*i/steps, diff_y*i/steps, diff_theta*i/steps)
            self.set_shape_states(glass_new_states, self.poured_glass_states)
            pyflex.step()
            tmp_states = glass_new_states

        # move water to target cup, wait for it to be stable
        pyflex.set_positions(particle_pos)
        pyflex.set_velocities(particle_vel)
        steps = 200
        for i in range(steps):
            pyflex.step()

    def set_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.state_goal = state_goal

    def get_goal(self):
        # print("get goal is called!")
        # print("self.state_goal: ", self.state_goal)
        # return {
        #     'desired_goal': self.state_goal,
        #     'state_desired_goal': self.state_goal,
        # }
        return self.state_dict_goal 

    def initialize_camera(self, make_multi_world_happy = None): 
        '''
        set the camera width, height, position and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        x_center = self.x_center # center of the glass floor
        z = self.fluid_params['z'] # lower corner of the water fluid along z-axis.
        self.camera_params = {
                        'pos': np.array([x_center + 1.5, 1.0 + 1.7, z + 0.2]),
                        'angle': np.array([0.45 * np.pi, -65/180. * np.pi, 0]),
                        # 'pos': np.array([x_center -1.3, 0.8, z + 0.5]),
                        # 'angle': np.array([0, 0, -0.5 * np.pi]),
                        'width': self.camera_width,
                        'height': self.camera_height
                        }

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        particle_pos = pyflex.get_positions().reshape((-1, self.dim_position))
        particle_vel = pyflex.get_velocities().reshape((-1, self.dim_velocity))
        particle_state = np.concatenate((particle_pos, particle_vel), axis = 1).reshape((1, -1))

        pouring_glass_state = np.array([self.glass_x, self.glass_y, 0, self.glass_rotation, self.glass_params['glass_dis_x'], 
            self.glass_params['glass_dis_z'], self.glass_params['height']])
        poured_glass_state = self.get_poured_glass_state()
        glass_state = np.concatenate((pouring_glass_state.reshape(1,-1), poured_glass_state.reshape(1,-1)), axis = 1)

        obs = np.concatenate((particle_state, glass_state), axis = 1)

        new_obs = dict(
            observation=obs,
            state_observation=obs,
            desired_goal=self.state_goal,
            state_desired_goal=self.state_goal,
            achieved_goal=obs,
            state_achieved_goal=obs,
        )

        return new_obs
