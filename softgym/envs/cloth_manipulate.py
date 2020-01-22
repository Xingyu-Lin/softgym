from gym.spaces import Box, Dict
import random
import os
import os.path as osp
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.core.multitask_env import MultitaskEnv
from softgym.envs.action_space import PickerPickPlace
import numpy as np
import copy
import pickle


class ClothManipulate(ClothFlattenEnv, MultitaskEnv):
    def __init__(self, cached_states_path='cloth_manipulate_init_states.pkl' ,**kwargs):
        '''
        Wrap cloth flatten to be goal conditioned cloth manipulation.
        The goal is a random cloth state.

        goal_num: how many goals to sample for each taks variation.
        '''

        ClothFlattenEnv.__init__(self, cached_states_path=cached_states_path, **kwargs)

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
        If the path exists, load from it. Should be a list of (config, states)
        :param cached_states_path:
        :return:
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

    def generate_env_variation(self, config, num_variations=4, goal_num=4, save_to_file=False):
        generated_configs, generated_init_states = ClothFlattenEnv.generate_env_variation(self, num_variations=num_variations)
        goal_dict = {}
        for idx in range(len(generated_configs)):
            ClothFlattenEnv.set_scene(self, generated_configs[idx], generated_init_states[idx])
            self.action_tool.reset()
            goals = self.sample_goals(goal_num)
            goal_dict[idx] = goals

        combined = (generated_configs, generated_init_states, goal_dict)
        with open(self.cached_states_path, 'wb') as handle:
            pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.cached_configs = generated_configs
        self.cached_init_states = generated_init_states
        self.cached_goal_dicts = goal_dict

    def sample_goals(self, batch_size=1):
        """
        just do some random actions on the cloth to generate a new state.
        """

        goal_observations = []

        initial_state = copy.deepcopy(self.get_state())
        for _ in range(batch_size):
            print("sample goals idx {}".format(_))
            self.set_state(initial_state)
            self._random_pick_and_place(pick_num=2)

            self._center_object()
            env_state = copy.deepcopy(self.get_state())
            goal = np.concatenate([env_state['particle_pos'], env_state['particle_vel'], env_state['shape_pos']])

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
        # print("shape of obs['state_achieved_goal']: {}, shape of obs['state_desired_goal']: {}".format(
        #     obs['state_achieved_goal'].shape, obs['state_desired_goal'].shape
        # ))
        r = -np.linalg.norm(obs['state_achieved_goal'] - obs['state_desired_goal'])
        return r

    def compute_rewards(self, action, obs, info=None):
        '''
        rewards in state space.
        '''
        # TODO: need to rename compute_reward / _get_obs in the super env's _step function.

        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        return -dist

    def _reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        ClothFlattenEnv._reset(self)
        self.resample_goals()

        return self._get_obs()

    def resample_goals(self):
        goal_idx = np.random.randint(len(self.cached_goal_dicts[self.current_config_id]["state_desired_goal"]))

        print("current config idx is {}, goal idx is {}".format(self.current_config_id, goal_idx))
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
        if self.state_goal is None:
            self.resample_goals()

        obs = obs.reshape((1, -1))
        new_obs = dict(
            observation=obs,
            state_observation=obs,
            desired_goal=self.state_goal[:, :len(obs[0])],
            state_desired_goal=self.state_goal[:, :len(obs[0])],
            achieved_goal=obs,
            state_achieved_goal=obs,
        )

        return new_obs

    def _get_obs(self):
        obs = ClothFlattenEnv._get_obs(self)
        return self._update_obs(obs)
