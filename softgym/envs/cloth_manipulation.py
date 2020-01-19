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

class ClothManipulation(ClothFlattenEnv, MultitaskEnv):
    def __init__(self, observation_mode, action_mode, goal_num=5, **kwargs):
        '''
        Wrap cloth flatten to be goal conditioned cloth manipulation.
        The goal is a random cloth state.

        goal_num: how many goals to sample for each taks variation.
        '''

        ClothFlattenEnv.__init__(
            self,
            observation_mode=observation_mode,
            action_mode=action_mode,
            **kwargs
        )
    
        self.goal_num = goal_num
        self.dict_goals = [None for _ in range(len(self.cached_configs))]
        self.state_goal = None

        # TODO: this is not correct now.
        self.obs_box = Box(np.array([-np.inf] * 2),
                                         np.array([np.inf] * 2), dtype=np.float32)
        self.goal_box = Box(np.array([-np.inf] * 2),
                                         np.array([np.inf] * 2), dtype=np.float32)


        self.observation_space = Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),
            ('achieved_goal', self.goal_box),
            ('state_achieved_goal', self.goal_box),
        ])

    def sample_goals(self, batch_size=1):
        """
        just do some random actions on the cloth to generate a new state.
        """

        goal_observations = []
        for _ in range(batch_size):
            # use a pikerpickplace to pick a point and drop it
            print("sample goals idx {}".format(_))

            max_wait_step = 300
            stable_vel_threshold = 0.01

            num_picker = 2
            picker = PickerPickPlace(num_picker = num_picker, particle_radius=0.05)
            
            action = np.zeros((num_picker, 6))
            first_particle_pos = pyflex.get_positions()[:3]
            last_particle_pos = pyflex.get_positions()[-4:-1]

            action[0, :3] = first_particle_pos
            action[1, :3] = last_particle_pos
            
            action[0, 3:] = copy.deepcopy(first_particle_pos)
            action[0, 4] = np.random.rand() * 5
            action[1, 3:] = copy.deepcopy(last_particle_pos)
            action[1, 4] = np.random.rand() * 5

            picker.step(action)

            stable_action = np.ones((num_picker, 6)) * -1
            stable_action[:, 1] = 4
            stable_action[:, 4] = 4
            picker.step(stable_action)

            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break

            self._center_object()
            env_state = copy.deepcopy(self.get_state())
            goal = np.concatenate([env_state['particle_pos'], env_state['particle_vel'], env_state['shape_pos']])

            goal_observations.append(goal)

        goal_observations = np.asarray(goal_observations).reshape((batch_size, -1))

        return {
            'desired_goal': goal_observations,
            'state_desired_goal': goal_observations,
        }

    def compute_reward(self, action, obs, set_prev_reward=False, info = None):
        '''
        reward is the l2 distance between the goal state and the current state.
        '''
        # print("shape of obs['state_achieved_goal']: {}, shape of obs['state_desired_goal']: {}".format(
        #     obs['state_achieved_goal'].shape, obs['state_desired_goal'].shape
        # ))
        r = -np.linalg.norm(
            obs['state_achieved_goal'] - obs['state_desired_goal'])
        return r

    def compute_rewards(self, action, obs, info = None):
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
        # if self.state_dict_goal is None: # NOTE: only suits for skewfit algorithm, because we are not actually sampling from this
        # true underlying env, but only sample from the vae latents. This reduces overhead to sample a goal each time for now.     
        ClothFlattenEnv._reset(self) 
        self.resample_goals(self.goal_num)

        return self._get_obs()

    def resample_goals(self, num=5):
        if self.dict_goals[self.current_config_idx] is None:
            self.dict_goals[self.current_config_idx] = self.sample_goals(num)

        goal_idx = np.random.randint(len(self.dict_goals[self.current_config_idx]["state_desired_goal"]))

        print("current config idx is {}, goal idx is {}".format(self.current_config_idx, goal_idx))
        self.dict_goal = {
            "desired_goal": self.dict_goals[self.current_config_idx]["desired_goal"][goal_idx], 
            "state_desired_goal": self.dict_goals[self.current_config_idx]["state_desired_goal"][goal_idx]
        } 

        self.state_goal = self.dict_goal['state_desired_goal'].reshape((1, -1)) # the real goal we want, np array

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
        # TODO: implement this
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
