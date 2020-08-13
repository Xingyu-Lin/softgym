from gym.spaces import Dict
import os.path as osp
import pyflex
from softgym.core.multitask_env import MultitaskEnv
from softgym.envs.rope_flatten_new import RopeFlattenNewEnv
import numpy as np
import copy
import pickle
from softgym.utils.pyflex_utils import center_object, random_pick_and_place

class RopeManipulateEnv(RopeFlattenNewEnv, MultitaskEnv):
    def __init__(self, goal_sampling_mode='fixed_goal', goal_num=10, cached_states_path='rope_manipulate_init_states.pkl', **kwargs):
        """
        Wrap rope flatten to be goal conditioned rope manipulation.
        The goal is a random rope state.
        """

        self.goal_num = goal_num
        self.goal_sampling_mode = goal_sampling_mode
        if self.goal_sampling_mode == 'fixed_goal':
            self.goal_num = 1
        RopeFlattenNewEnv.__init__(self, cached_states_path=cached_states_path, **kwargs)

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

    def generate_env_variation(self, num_variations=4, save_to_file=False):
        generated_configs, generated_init_states = RopeFlattenEnv.generate_env_variation(self, num_variations=num_variations)
        goal_dict = {}
        goal_num = self.goal_num
        for idx in range(len(generated_configs)):
            if self.goal_sampling_mode == 'fixed_goal':  # for better generating the goal states
                RopeFlattenEnv.set_scene(self, generated_configs[idx])
                self.action_tool.reset([0., -1., 0.])
                goals = self.sample_goals(goal_num)
            else:
                RopeFlattenEnv.set_scene(self, generated_configs[idx], generated_init_states[idx])
                self.action_tool.reset([0., -1., 0.])
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
        if self.goal_sampling_mode != 'fixed_goal':
            for _ in range(batch_size):
                print("sample goals idx {}".format(_))
                self.set_state(initial_state)
                random_pick_and_place(pick_num=2)

                center_object()
                env_state = copy.deepcopy(self.get_state())
                goal = np.concatenate([env_state['particle_pos'], env_state['particle_vel'], env_state['shape_pos']])

                goal_observations.append(goal)
        else:
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[:, 1] = 0.05  # Set the rope flatten on the ground. Assume that the particle radius is 0.05
            pyflex.set_positions(curr_pos)

            center_object()
            env_state = copy.deepcopy(self.get_state())
            goal = np.concatenate([env_state['particle_pos'], env_state['particle_vel'], env_state['shape_pos']])
            for _ in range(batch_size):
                goal_observations.append(goal)

        goal_observations = np.asarray(goal_observations).reshape((batch_size, -1))

        return {
            'desired_goal': goal_observations,
            'state_desired_goal': goal_observations,
        }

    def compute_reward(self, action, obs, set_prev_reward=False, info=None):
        """reward is the l2 distance between the goal state and the current state."""
        # print("shape of obs['state_achieved_goal']: {}, shape of obs['state_desired_goal']: {}".format(
        #     obs['state_achieved_goal'].shape, obs['state_desired_goal'].shape
        # ))
        r = -np.linalg.norm(
            obs['state_achieved_goal'] - obs['state_desired_goal'])
        return r

    def compute_rewards(self, action, obs, info=None):
        """
        rewards in state space.
        """
        # TODO: need to rename compute_reward / _get_obs in the super env's _step function.

        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        return -dist

    def _reset(self):
        """
        reset to environment to the initial state.
        return the initial observation.
        """
        self.resample_goals()
        return RopeFlattenEnv._reset(self)

    def resample_goals(self):
        goal_idx = np.random.randint(len(self.cached_goal_dicts[self.current_config_id]["state_desired_goal"]))

        # print("current config idx is {}, goal idx is {}".format(self.current_config_id, goal_idx))
        self.dict_goal = {
            "desired_goal": self.cached_goal_dicts[self.current_config_id]["desired_goal"][goal_idx],
            "state_desired_goal": self.cached_goal_dicts[self.current_config_id]["state_desired_goal"][goal_idx]
        }

        self.state_goal = self.dict_goal['state_desired_goal'].reshape((1, -1))  # the real goal we want, np array

    def get_env_state(self):
        """
        get the postion, velocity of flex particles, and postions of flex shapes.
        a wrapper to be compatiable with the MultiTask Env.
        """
        return self.get_state()

    def set_env_state(self, state_dic):
        """
        set the postion, velocity of flex particles, and postions of flex shapes.
        a wrapper to be compatiable with the MultiTask Env.
        """
        return self.set_state(state_dic)

    def set_to_goal(self, goal):
        """
        given a goal, set the flex state to be that goal.
        needed by image env to sample goals.
        """
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
        """
        return the observation based on the current flex state.
        """

        obs = obs.reshape((1, -1))
        if self.observation_mode == 'point_cloud':
            n = pyflex.get_n_particles()
            goal = np.zeros(self.particle_obs_dim)  # all particle positions
            goal_particle_pos = self.state_goal[0][:n * 4].reshape([-1, 4])[:, :3]
            goal_particle_pos = goal_particle_pos.flatten()
            goal[:len(goal_particle_pos)] = goal_particle_pos

            if self.action_mode in ['sphere', 'picker']:  # should just set as random shape positions
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                shape_pos = shapes[:, :3].flatten()
                goal = np.concatenate([goal.flatten(), np.zeros_like(shape_pos)])

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
        obs = RopeFlattenEnv._get_obs(self)
        return self._update_obs(obs)
