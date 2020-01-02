from gym.spaces import Box, Dict
import random
import os
import os.path as osp
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenPointControlEnv
from softgym.core.multitask_env import MultitaskEnv
import numpy as np

class ClothFlattenPointControlGoalConditionedEnv(ClothFlattenPointControlEnv, MultitaskEnv):
    def __init__(self, observation_mode, action_mode, horizon=100, headless=False, render=True, render_mode='particle'):
        '''
        Wrap cloth flatten to be goal conditioned
        '''
        self.state_dict_goal = None
        ClothFlattenPointControlEnv.__init__(
            self,
            observation_mode=observation_mode,
            action_mode=action_mode,
            horizon=horizon,
            render=render,
            headless=headless,
            render_mode=render_mode
        )

        self.dim_position = 4
        self.dim_velocity = 3
        self.dim_shape_state = 14
        self.particle_num = pyflex.get_n_particles()
        # TODO: we might want to add positions of the sphere to the observation space
        self.obs_box = Box(np.array([-np.inf] * pyflex.get_n_particles()),
                                         np.array([np.inf] * pyflex.get_n_particles()), dtype=np.float32)
        self.goal_box = Box(np.array([-np.inf] * pyflex.get_n_particles()),
                                         np.array([np.inf] * pyflex.get_n_particles()), dtype=np.float32)

        self.observation_space = Dict([
            ('observation', self.obs_box),
            ('state_observation', self.obs_box),
            ('desired_goal', self.goal_box),
            ('state_desired_goal', self.goal_box),
            ('achieved_goal', self.goal_box),
            ('state_achieved_goal', self.goal_box),
        ])

    def sample_goals(self, batch_size):
        # TODO: just change the position of the cloth should be fine.

        # make the cloth lying on the ground
        cloth_pos = np.ones((self.particle_num, self.dim_position))
        cloth_vel = np.zeros((self.particle_num, self.dim_velocity))
        
        initX = self.params[0]
        initY = self.params[1]
        initZ = self.params[2]

        lower = np.zeros((1, 3)) # y should always be 0, which means exactly on the ground
        lower = np.array([initX, -initY, initZ]).reshape(1,-1) # lower is just the same as the scene's reset lower
        lower[0][0] += np.random.uniform(-0.2, 0.2)
        lower[0][2] += np.random.uniform(-0.2, 0.2)
        
        # cloth goal: cloth flattened on the ground. Position is determined by lower.
        dimx = int(self.params[3])
        dimy = int(self.params[4])
        dimz = 1
        # print("dimx: {}, dimy: {}, dimz: {}".format(dimx, dimy, dimz))
        cnt = 0
        radius = 0.05
        for z in range(dimz):
            for y in range(dimy):
                for x in range(dimx):
                    cloth_pos[cnt][:3] = lower + np.array([x, z, y])*radius #TODO: make radius a changable parameter that we pass to the scene
                    cnt += 1

        # cnt = 0
        # for x in range(self.fluid_params['dim_x']):
        #     for y in range(self.fluid_params['dim_y']):
        #         for z in range(self.fluid_params['dim_z']):
        #             cloth_pos[cnt][:3] = lower + np.array([x, y, z])*self.fluid_params['radius'] / 2#+ np.random.rand() * 0.01
        #             cnt += 1

        cloth_goal = np.concatenate((cloth_pos.reshape(1, -1), cloth_vel.reshape(1, -1)), axis = 1)

        # gripper goal: just stay above the lower corner + random distance
        if self.action_mode.startswith('sphere'):
            sphere_lower = lower
            sphere_lower[0][1] = 0.2 # y distance is fixed
            sphere_lower[0][0] += np.random.rand()
            sphere_lower[0][2] += np.random.rand()
            sphere_distance = np.array([0.2 + np.random.rand()]).reshape(1,1)
            goal = np.concatenate([cloth_goal, sphere_lower, sphere_distance], axis = 1)
        else:
            goal = cloth_goal

        self.state_goal = goal

        return {
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def compute_reward(self, action, obs, info = None):
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

    def reset(self, dropPoint=1000, xdim=64, ydim=32):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        # if self.state_dict_goal is None: # NOTE: only suits for skewfit algorithm, because we are not actually sampling from this
        # true underlying env, but only sample from the vae latents. This reduces overhead to sample a goal each time for now.
        self.state_dict_goal = self.sample_goal()        
        ClothFlattenPointControlEnv.reset(self, dropPoint, xdim, ydim) # TODO: Jake's reset does not return observations

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
        # TODO: implement this
        state_goal = goal['state_desired_goal']
        particle_pos = state_goal[:self.particle_num * self.dim_position]
        particle_vel = state_goal[self.particle_num * self.dim_position: (self.dim_position + self.dim_velocity) * self.particle_num]
        
        # move cloth to target position, wait for it to be stable
        pyflex.set_positions(particle_pos)
        pyflex.set_velocities(particle_vel)
        steps = 20
        for _ in range(steps):
            pyflex.step()

        # move gripper to target position.
        # the idea is to use the spherestep function to gradually move the grippers to the target position.
        if self.action_mode.startswith('sphere'):
            tmp = (self.dim_position + self.dim_velocity) * self.particle_num
            gripper_x, gripper_y, gripper_z, gripper_distance = \
                    state_goal[tmp], state_goal[tmp + 1], state_goal[tmp+2], state_goal[tmp+3]     

            init_dist = self.prev_dist[0]
            init_middle = self.prev_middle[0]

            steps = 50
            diff_dist = (init_dist - gripper_distance) / steps
            pos_diff_x = (gripper_x - init_middle[0]) / steps
            pos_diff_y = (gripper_y - init_middle[1]) / steps
            pos_diff_z = (gripper_z - init_middle[2]) / steps

            # print("pos_diff_x: {}, pos_diff_y: {}, pos_diff_z: {}".format(pos_diff_x, pos_diff_y, pos_diff_z))
            # print("diff_dist: {}".format(diff_dist))

            for _ in range(steps):
                last_states = pyflex.get_shape_states()
                self.sphereStep(np.array([pos_diff_x, pos_diff_y, pos_diff_z, 0, diff_dist] * 2), last_states)
                pyflex.step()
        

    def set_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.state_goal = state_goal

    def get_goal(self):
        return self.state_dict_goal 

    def initialize_camera(self, make_multiworld_happy = None):
        '''
        set the camera width, height, ition and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        # TODO: might be different
        self.camera_params = {
            'pos': np.array([-0.2, 4., 4]),
            'angle': np.array([0., -45 / 180. * np.pi, 0.]),
            'width': self.camera_width,
            'height': self.camera_height
        }

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        obs = ClothFlattenPointControlEnv._get_obs(self)

        obs = obs.reshape((1, -1))
        new_obs = dict( # TODO: jake should include gripper position, and cloth particle velocity into the state space.
            observation=obs,
            state_observation=obs,
            desired_goal=self.state_goal[:, :len(obs[0])],
            state_desired_goal=self.state_goal[:, :len(obs[0])],
            achieved_goal=obs,
            state_achieved_goal=obs,
        )

        return new_obs  

