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
import pickle


class PourWaterPosControlGoalConditionedEnv(PourWaterPosControlEnv, MultitaskEnv):
    def __init__(self, goal_sampling_mode='fixed_goal', cached_states_path='pour_water_multitask_init_states.pkl', **kwargs):
        '''
        This class implements a multi-goal pouring water task.
        Where the goal has different positions of the pouring glass.
        '''

        self.goal_sampling_mode = goal_sampling_mode
        PourWaterPosControlEnv.__init__(self, cached_states_path=cached_states_path, **kwargs)
        self.state_goal = None

        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('state_observation', self.observation_space),
            ('desired_goal', self.observation_space),
            ('state_desired_goal', self.observation_space),
            ('achieved_goal', self.observation_space),
            ('state_achieved_goal', self.observation_space),
        ])

    def get_cached_configs_and_states(self, cached_states_path):
        """
        If the path exists, load from it. Should be a list of (config, states, goals)
        """
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            cached_states_path = osp.join(cur_dir, cached_states_path)
        if not osp.exists(cached_states_path):
            return False
        with open(cached_states_path, "rb") as handle:
            self.cached_configs, self.cached_init_states, self.cached_goal_dicts = pickle.load(handle)
        print('{} config, state and goal pairs loaded from {}'.format(len(self.cached_init_states), cached_states_path))
        return True

    def generate_env_variation(self, config, num_variations=5, goal_num=10, save_to_file=False):
        generated_configs, generated_init_states = PourWaterPosControlEnv.generate_env_variation(self, 
            config, num_variations=num_variations)
        goal_dict = {}
        for idx in range(len(generated_configs)):
            PourWaterPosControlEnv.set_scene(self, generated_configs[idx], generated_init_states[idx])
            if self.goal_sampling_mode == 'fixed_goal':
                goals = self.sample_goals(1)
            else:
                goals = self.sample_goals(goal_num)
            goal_dict[idx] = goals

        combined = (generated_configs, generated_init_states, goal_dict)
        with open(self.cached_states_path, 'wb') as handle:
            pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.cached_configs = generated_configs
        self.cached_init_states = generated_init_states
        self.cached_goal_dicts = goal_dict

    def sample_goals(self, batch_size=1):
        '''
        The goal is to have all the water particles set to be in the target cup.
        The pouring cup could be at different positions.
        '''
        goal_observations = []

        initial_state = copy.deepcopy(self.get_state())
        for idx in range(batch_size):
            print("sample goals idx {}".format(idx))
            self.set_state(initial_state)

            fluid_pos = np.ones((self.particle_num, self.dim_position))

            fluid_radius = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            fluid_dis = np.array([1.2 * fluid_radius, fluid_radius * 0.45, 1.2 * fluid_radius])
            lower_x = self.glass_params['poured_glass_x_center'] - self.glass_params['poured_glass_dis_x'] / 2.
            lower_z = -self.glass_params['poured_glass_dis_z'] / 2 + 0.05
            lower_y = self.glass_params['poured_border']
            lower = np.array([lower_x, lower_y, lower_z])
            cnt = 0
            for x in range(self.fluid_params['dim_x']):
                for y in range(self.fluid_params['dim_y']):
                    for z in range(self.fluid_params['dim_z']):
                        fluid_pos[cnt][:3] = lower + np.array([x, y, z]) * fluid_dis  # + np.random.rand() * 0.01
                        cnt += 1
            
            pyflex.set_positions(fluid_pos)
            
            # make control cup hang near the target cup and rotates towards the target cup, simulating a real pouring.
            if self.goal_sampling_mode != 'fixed_goal':
                pouring_glass_x = lower_x -  (0.9 + np.random.rand() * 0.2) * self.glass_params['glass_distance']
                pouring_glass_y = (1.5 + np.random.rand() * 0.4) * self.glass_params['poured_height']
                pouring_theta = 0.8 + np.random.rand() * 0.4 * np.pi
            else: # fixed goal
                pouring_glass_x = lower_x -  0.8 * self.glass_params['glass_distance']
                pouring_glass_y = 1.8 * self.glass_params['poured_height']
                pouring_theta = 0.6 * np.pi
            control_cup_x, control_cup_y, control_cup_theta = pouring_glass_x,  pouring_glass_y, pouring_theta

            # move controled cup tp target position
            tmp_states = self.glass_states
            diff_x = control_cup_x - self.glass_x
            diff_y = control_cup_y - self.glass_y
            diff_theta = control_cup_theta - self.glass_rotation
            steps = 300
            for i in range(steps):
                glass_new_states = self.rotate_glass(tmp_states, diff_x * i / steps, diff_y * i / steps, diff_theta * i / steps)
                self.set_shape_states(glass_new_states, self.poured_glass_states)
                pyflex.step()
                tmp_states = glass_new_states

            particle_pos = pyflex.get_positions().reshape((1, -1))
            particle_vel = pyflex.get_velocities().reshape((1, -1))
            shape_pos = pyflex.get_shape_states().reshape((1, -1))

            water_height = self._get_current_water_height()
            # print("water height: ", water_height)
            cup_state = np.array([control_cup_x, control_cup_y, control_cup_theta, self.glass_dis_x, self.glass_dis_z, self.height,
                                  self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                  water_height]).reshape((1, -1))
            goal = np.concatenate([particle_pos, particle_vel, shape_pos, cup_state], axis=1)

            if self.goal_sampling_mode != 'fixed_goal':
                goal_observations.append(goal)
            else: # fixed goal.
                for _ in range(batch_size):
                    goal_observations.append(goal)
                break


        goal_observations = np.asarray(goal_observations).reshape((batch_size, -1))
        return {
            'desired_goal': goal_observations,
            'state_desired_goal': goal_observations,
        }

    def compute_reward(self, action, obs, set_prev_reward=False, info=None):
        '''
        reward is the l2 distance between the goal state and the current state.
        '''
        # print("obs is", obs)
        r = -np.linalg.norm(
            obs['state_achieved_goal'] - obs['state_desired_goal'])
        return r

    def compute_rewards(self, action, obs, set_prev_reward=False, info=None):
        '''
        rewards in state space.
        '''
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        return -dist

    def _reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        self.resample_goals()
        return PourWaterPosControlEnv._reset(self)
        
    def resample_goals(self):
        goal_idx = np.random.randint(len(self.cached_goal_dicts[self.current_config_id]["state_desired_goal"]))

        # print("current config idx is {}, goal idx is {}".format(self.current_config_id, goal_idx))
        self.dict_goal = {
            "desired_goal": self.cached_goal_dicts[self.current_config_id]["desired_goal"][goal_idx],
            "state_desired_goal": self.cached_goal_dicts[self.current_config_id]["state_desired_goal"][goal_idx]
        }

        self.state_goal = self.dict_goal['state_desired_goal'].reshape((1, -1))  # the real goal we want, np array

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
        particle_pos = state_goal[:self.particle_num * self.dim_position]
        particle_vel = state_goal[self.particle_num * self.dim_position: (self.dim_position + self.dim_velocity) * self.particle_num]
        shape_pos = state_goal[(self.dim_position + self.dim_velocity) * self.particle_num:-10]

        # move cloth to target position, wait for it to be stable
        pyflex.set_positions(particle_pos)
        pyflex.set_velocities(particle_vel)
        pyflex.set_shape_states(shape_pos)
        steps = 5
        for _ in range(steps):
            pyflex.step()

    def set_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.state_goal = state_goal

    def get_goal(self):
        return self.dict_goal

    def _update_obs(self, obs):
        '''
        return the observation based on the current flex state.
        '''

        obs = obs.reshape((1, -1))

        if self.observation_mode == 'point_cloud':
            goal = np.zeros(shape=self.particle_obs_dim + 11, dtype=np.float)
            n = pyflex.get_n_particles()

            goal_particle_pos = self.state_goal[0][:n*4].reshape([-1, 4])[:, :3]
            goal_water_height = np.max(goal_particle_pos[:, 1])
            goal_particle_pos = goal_particle_pos.flatten()
            goal[:len(goal_particle_pos)] = goal_particle_pos
            goal[-11:] = self.state_goal[0, -11:] # cup_state.
            goal = goal.reshape((1, -1))

        new_obs = dict(
            observation=obs,
            state_observation=obs,
            desired_goal=goal,
            state_desired_goal=goal,
            achieved_goal=obs,
            state_achieved_goal=obs,
        )

        return new_obs

    def _get_obs(self):
        obs = PourWaterPosControlEnv._get_obs(self)
        return self._update_obs(obs)

