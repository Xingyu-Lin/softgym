import numpy as np
from gym.spaces import Box, Dict

import pyflex
from softgym.envs.fluid_env import FluidEnv
from softgym.envs.pass_water import PassWater1DEnv
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


class PassWater1DGoalConditionedEnv(PassWater1DEnv, MultitaskEnv):
    def __init__(self, goal_sampling_mode='fixed_goal', cached_states_path='pass_water_multitask_init_states.pkl', **kwargs):
        '''
        This class implements a single-goal passing water task.
        The single goal is to have the glass exactly at the terminal position.
        This is mainly used for testing RIG, when there is only one single goal.
        '''

        self.goal_sampling_mode = goal_sampling_mode
        PassWater1DEnv.__init__(self, cached_states_path=cached_states_path, **kwargs)
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
        generated_configs, generated_init_states = PassWater1DEnv.generate_env_variation(self, 
            config, num_variations=num_variations)
        goal_dict = {}
        for idx in range(len(generated_configs)):
            PassWater1DEnv.set_scene(self, generated_configs[idx], generated_init_states[idx])
            print("generating goal for config: ", idx)
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
        The goal is to move the glass at the terminal position.
        We only have one goal for each init state, for now.
        '''
        goal_observations = []

        # if self.goal_sampling_mode == 'fixed_goal':
        #     assert batch_size == 1

        # initial_state = copy.deepcopy(self.get_state())
        # # for idx in range(batch_size):
        #     # print("sample goals idx {}".format(idx))
        #     self.set_state(initial_state)

        # move controled cup to target position
        tmp_states = self.glass_states
        dx = 0.005
        while True:
            if self.terminal_x - tmp_states[0,0] < 0.002:
                dx = self.terminal_x - tmp_states[0,0]
            glass_new_states = self.move_glass(tmp_states, dx + tmp_states[0, 0])
            pyflex.set_shape_states(glass_new_states)
            pyflex.step()
            tmp_states = glass_new_states 
            if np.abs(self.terminal_x - tmp_states[0,0]) < 1e-3:
                break

        particle_pos = pyflex.get_positions().reshape((1, -1))
        particle_vel = pyflex.get_velocities().reshape((1, -1))
        shape_pos = pyflex.get_shape_states().reshape((1, -1))
        goal = np.concatenate([particle_pos, particle_vel, shape_pos], axis=1)
        print("generated goal length: ", len(goal))
        for i in range(batch_size):
            goal_observations.append(goal)

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
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        return -dist

    def _reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''

        self.resample_goals()
        return PassWater1DEnv._reset(self)

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
        shape_pos = state_goal[(self.dim_position + self.dim_velocity) * self.particle_num:]

        # move cup to target position, wait for it to be stable
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
        n = pyflex.get_n_particles()
        # print("_update_obs: current config idx is ", self.current_config_id)
        # print("_update_obs: num of particles in scene: ", n)
        # print("_update_obs: state goal shape: ", self.state_goal.shape)
        goal = np.zeros(n * 3 + 1) # all particle positions and glass x
        for i in range(n):
            goal[i*3: (i+1)*3] = self.state_goal[0, i*4: i*4 + 3]
        goal[-1] = self.state_goal[0, 7*n] # 7*n is the first idx for shape states, and that is the x of the floor box.
        # assert np.abs(goal[-1] - self.terminal_x) < 1e-5
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
        obs = PassWater1DEnv._get_obs(self)
        return self._update_obs(obs)

    def _get_info(self):
        reward = PassWater1DEnv.compute_reward(self,set_prev_reward=True)
        return dict(real_task_reward=reward)
