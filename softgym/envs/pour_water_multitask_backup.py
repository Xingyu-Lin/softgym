import numpy as np
from gym.spaces import Box, Dict

import pyflex
from softgym.envs.fluid_env import FluidEnv 
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

class PourWaterPosControlGoalConditionedEnv(FluidEnv, MultitaskEnv):
    def __init__(self, observation_mode = 'full_state', action_mode = 'direct', render = True, headless = False, 
            horizon = 300, deterministic = True, render_mode = 'particle'):
        '''
        This class implements a pouring water task.
        
        observation_mode: "cam_rgb" or "full_state"
        action_mode: "direct"
        horizon: environment horizon
        
        TODO: add more description of the task.
        TODO: allow parameter configuring of the scence.
        '''

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.wall_num = 5 # number of glass walls. floor/left/right/front/back 

        # debug usage
        self.camera_called_time = 0
        self.state_dict_goal = None

        FluidEnv.__init__(self, horizon = horizon, deterministic = deterministic, render_mode = render_mode, render = render, 
            headless=headless)
        assert observation_mode in ['full_state'] 
        assert action_mode in ['direct'] 

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

        if action_mode == 'direct':
            self.action_direct_dim = 3 # control the (x, y) corrdinate of the floor center, and theta its rotation angel.
            self.action_space = Box(np.array([-0.05] * self.action_direct_dim), np.array([0.05] * self.action_direct_dim), dtype=np.float32) # space range: no larger than cup border
        else:
            raise NotImplementedError

        ### goal related vars

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
        lower_x = self.glass_params['poured_glass_x_center'] - self.glass_params['poured_glass_dis_x'] / 2.
        lower_z = -self.glass_params['poured_glass_dis_z'] / 2.
        lower_y = self.glass_params['poured_border'] 
        lower = np.array([lower_x, lower_y, lower_z])
        # print("in sample goals, fluid lower x {} lower y {} lower z {}".format(lower_x, lower_y, lower_z))
        cnt = 0
        for x in range(self.fluid_params['dim_x']):
            for y in range(self.fluid_params['dim_y']):
                for z in range(self.fluid_params['dim_z']):
                    fluid_pos[cnt][:3] = lower + np.array([x, y, z])*self.fluid_params['radius'] / 3 #+ np.random.rand() * 0.01
                    cnt += 1

        fluid_goal = np.concatenate((fluid_pos.reshape(1, -1), fluid_vel.reshape(1, -1)), axis = 1)

        # the full goal state includes the control cup and target cup's state
        # make control cup hang near the target cup and rotates towards the target cup, simulating a real pouring.
        pouring_glass_x = lower_x - self.glass_params['poured_height']
        pouring_glass_z = 0.
        pouring_glass_y = 2.2 * self.glass_params['poured_height']
        pouring_theta = 0.6 * np.pi
        # print("in sampling goal set control cup x {} control cup y {} control cup theta {}".format(
        #     pouring_glass_x, pouring_glass_y, pouring_theta
        # ))

        pouring_glass_goal = np.array([pouring_glass_x, pouring_glass_y, pouring_glass_z, pouring_theta, self.glass_params['glass_dis_x'], 
            self.glass_params['glass_dis_z'], self.glass_params['height']])
        poured_glass_goal = self.get_poured_glass_state()
        glass_goal = np.concatenate((pouring_glass_goal.reshape(1,-1), poured_glass_goal.reshape(1,-1)), axis = 1)

        goal = np.concatenate((fluid_goal, glass_goal), axis = 1)
        # print("goal.shape:", goal.shape )

        # print("goal :", goal)
        self.state_goal = goal
        # print("state goal: ", self.state_goal)

        return {
            'desired_goal': goal,
            'state_desired_goal': goal,
        }

    def get_poured_glass_state(self):
        return np.array([self.glass_params['poured_glass_x_center'], self.glass_params['poured_border'], 0, 0,
            self.glass_params['poured_glass_dis_x'], self.glass_params['poured_glass_dis_z'], self.glass_params['poured_height']])


    def compute_reward(self, action, obs, info = None):
        # print(obs)
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
        
    def reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        self.set_env_state(self.init_flex_state)
        if self.state_dict_goal is None: # only suits for skewfit algorithm, because we are not actually sampling from this
            # true underlying env, but only sample from the vae latents
            self.state_dict_goal = self.sample_goal()        
        return self._get_obs()
    
    def get_env_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
            'glass_x': self.glass_x, 'glass_y': self.glass_y, 'glass_rotation': self.glass_rotation, 'glass_states': self.glass_states}

    def set_env_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        pyflex.set_shape_states(state_dic["shape_pos"])
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        self.glass_x = state_dic['glass_x']
        self.glass_y = state_dic['glass_y']
        self.glass_rotation = state_dic['glass_rotation']
        self.glass_states = state_dic['glass_states']
        pyflex.step()
    
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

        pyflex.set_positions(particle_pos)
        pyflex.set_velocities(particle_vel)
        steps = 200
        for i in range(steps):
            # print("in set_to_goal, pyflex step {}".format(i))
            pyflex.step()
            # time.sleep(3)

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
        # print("intialize camera called time: {}".format(self.camera_called_time))
        x_center = self.x_center # center of the glass floor
        z = self.fluid_params['z'] # lower corner of the water fluid along z-axis.
        self.camera_params = {
                        'pos': np.array([x_center + 1.5, 0.8 + 1.7, z + 0.3]),
                        'angle': np.array([0.4 * np.pi, -65/180. * np.pi, 0]),
                        # 'pos': np.array([x_center -1.3, 0.8, z + 0.5]),
                        # 'angle': np.array([0, 0, -0.5 * np.pi]),
                        'width': self.camera_width,
                        'height': self.camera_height
                        }
        self.camera_called_time += 1

    def sample_glass_params(self, config):
        params = {}
        params['border_range'] = 0.015, 0.025
        params['height_range'] = 0.5, 0.7
        params['glass_distance_range'] = 0.5, 0.8
        params['poured_border_range'] = 0.015, 0.025
        params['poured_height_range'] = 0.5, 0.7

       
        params['border'] = self.rand_float(params['border_range'][0], params['border_range'][1]) # the thickness of the glass wall.
        params['height'] = self.rand_float(params['height_range'][0], params['height_range'][1]) # the height of the glass.
        params['glass_distance'] = self.rand_float(params['glass_distance_range'][0], params['glass_distance_range'][1]) # distance between the pouring glass and the poured glass
        params['poured_border'] = self.rand_float(params['poured_border_range'][0], params['poured_border_range'][1])
        params['poured_height'] = self.rand_float(params['poured_height_range'][0], params['poured_height_range'][1])
        if self.deterministic:
            for k in config:
                params[k] = config[k]

        self.border = params['border']
        self.height = params['height']
        self.glass_distance = params['glass_distance']
        self.poured_border = params['poured_border']
        self.poured_height = params['poured_height']

        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        if not self.deterministic:
            self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + self.rand_float(0., 0.1) # glass floor length
            self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + self.rand_float(0, 0.1) # glass width
            self.poured_glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + self.rand_float(0., 0.1) # glass floor length
            self.poured_glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + self.rand_float(0, 0.1) # glass width
        else:
            self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.1 # glass floor length
            self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.1 # glass width
            self.poured_glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.05# glass floor length
            self.poured_glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.05# glass width

        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['poured_glass_dis_x'] = self.poured_glass_dis_x
        params['poured_glass_dis_z'] = self.poured_glass_dis_z
        params['glass_x_center'] = self.x_center
        params['poured_glass_x_center'] = self.x_center + params['glass_distance']

        self.glass_params = params

    def set_scene(self):
        '''
        Construct the pouring water scence.
        '''
        # create fluid
        config_dir = osp.dirname(osp.abspath(__file__))
        config = open(osp.join(config_dir, "PourWaterDefaultConfig.yaml"), 'r')
        config = yaml.load(config)
        if self.deterministic:
            FluidEnv.set_scene(self, config["fluid"])
            print(config)
        else:
            FluidEnv.set_scene(self)

        # compute glass params
        self.sample_glass_params(config["glass"])

        # create pouring glass
        glass = self.create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)
        for i in range(len(glass)):
            halfEdge = glass[i][0]
            center = glass[i][1]
            quat = glass[i][2]
            pyflex.add_box(halfEdge, center, quat)

        # create poured glass
        poured_glass = self.create_glass(self.poured_glass_dis_x, self.poured_glass_dis_z, self.poured_height, self.poured_border)
        for i in range(len(poured_glass)):
            halfEdge = poured_glass[i][0]
            center = poured_glass[i][1]
            quat = poured_glass[i][2]
            pyflex.add_box(halfEdge, center, quat)
        
        # move pouring glass to be at ground
        self.glass_floor_centerx = self.x_center
        self.glass_states = self.init_glass_state(self.x_center, 0, self.glass_dis_x, self.glass_dis_z, self.height, self.border)

        # move poured glass to be at ground
        self.poured_glass_states = self.init_glass_state(self.x_center + self.glass_distance, 0, 
            self.poured_glass_dis_x, self.poured_glass_dis_z, self.poured_height, self.poured_border)

        
        self.set_shape_states(self.glass_states, self.poured_glass_states)

        # give some time for water to stablize 
        for i in range(150):
            pyflex.step()

        # record glass floor center x, y, and rotation
        self.glass_x = self.x_center
        self.glass_y = 0
        self.glass_rotation = 0

        self.init_flex_state = self.get_env_state()

        print("pour water inital scene constructed over...")

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

    def _step(self, action):
        '''
        action: np.ndarray of dim 1x3, (x, y, theta). (x, y) specifies the floor center coordinate, and theta 
            specifies the rotation.
        
        return: gym-like (next_obs, reward, done, info)
        '''

        # make action as increasement
        move = action[:2]
        rotate = action[2]
        move = np.clip(move, a_min = -self.border, a_max = self.border)
        rotate = np.clip(rotate, a_min = -self.border, a_max = self.border)
        dx, dy, dtheta = move[0], move[1], rotate
        x, y, theta = self.glass_x + dx, self.glass_y + dy, self.glass_rotation + dtheta
        y = max(0, y)

        info = {}

        # check if the movement of the pouring glass collide with the poured glass.
        new_states = self.rotate_glass(self.glass_states, x, y, theta)
        if not self.judge_glass_collide(new_states, theta):
            self.glass_states = new_states
            self.glass_x, self.glass_y, self.glass_rotation = x, y, theta
        else:
            # info["glass_collide"] = True
            pass

        # pyflex takes a step to update the glass and the water fluid
        self.set_shape_states(self.glass_states, self.poured_glass_states)
        if self.record_video:
            pyflex.step(capture = 1, path = self.video_path + 'render_' + str(self.time_step) + '.tga')
        else:
            pyflex.step() 

        # get reward and new observation for the agent.
        obs = self._get_obs()
        reward = self.compute_reward(action, obs)

        self.time_step += 1

        done = True if self.time_step == self.horizon else False
        return obs, reward, done, info

    def create_glass(self, glass_dis_x, glass_dis_z, height, border):
        """
        the glass is a box, with each wall of it being a very thin box in Flex.
        each wall of the real box is represented by a box object in FLex with really small thickness (determined by the param border)
        dis_x: the length of the glass
        dis_z: the width of the glass
        height: the height of the glass.
        border: the thickness of the glass wall.

        the halfEdge determines the center point of each wall.
        Note: this is merely setting the length of each dimension of the wall, but not the actual position of them.
        That's why left and right walls have exactly the same params, and so do front and back walls.   
        """
        center = np.array([0., 0., 0.])
        quat = self.quatFromAxisAngle([0, 0, -1.], 0.) 
        boxes = []

        # floor
        halfEdge = np.array([glass_dis_x/2. + border, border/2., glass_dis_z/2. + border])
        boxes.append([halfEdge, center, quat])

        # left wall
        halfEdge = np.array([border/2., (height)/2., glass_dis_z/2. + border])
        boxes.append([halfEdge, center, quat])

        # right wall
        boxes.append([halfEdge, center, quat])

        # back wall
        halfEdge = np.array([(glass_dis_x)/2., (height)/2., border/2.])
        boxes.append([halfEdge, center, quat])

        # front wall
        boxes.append([halfEdge, center, quat])

        return boxes

    def rotate_glass(self, prev_states, x, y, theta):
        '''
        given the previous states of the glass, rotate it with angle theta.
        update the states of the 5 boxes that form the box: floor, left/right wall, back/front wall. 
        rotate the glass, where the center point is the center of the floor (bottom wall).
        
        state:
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat 
        10-14: previous quat 
        '''
        dis_x, dis_z = self.glass_dis_x, self.glass_dis_z
        quat_curr = self.quatFromAxisAngle([0, 0, -1.], theta) 

        border = self.border

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        for i in range(5):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]

        x_center = x

        # rotation center is the floor center
        rotate_center = np.array([x_center, y, 0.])

        # floor: center position does not change
        states[0, :3] = np.array([x_center, y, 0.])

        # left wall: center must move right and move down. 
        relative_coord = np.array([-(dis_x+border)/2., (self.height+border)/2., 0.])
        states[1, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle = theta, relative=relative_coord)

        # right wall
        relative_coord = np.array([(dis_x+border)/2., (self.height+border)/2., 0.])
        states[2, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle = theta, relative=relative_coord)
        
        # back wall
        relative_coord = np.array([0, (self.height+border)/2., -(dis_z+border)/2.])
        states[3, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle = theta, relative=relative_coord)

        # front wall
        relative_coord = np.array([0, (self.height+border)/2., (dis_z+border)/2.])
        states[4, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle = theta, relative=relative_coord)

        states[:, 6:10] = quat_curr

        return states

    def init_glass_state(self, x, y, glass_dis_x, glass_dis_z, height, border, theta = 0., y_last = 0., theta_last = None):
        '''
        set the initial state of the glass.
        '''
        dis_x, dis_z = glass_dis_x, glass_dis_z
        x_center, y_curr, y_last  = x, y, y_last
        # print("in init_glass_state, y_curr is: ", y_curr)
        quat = self.quatFromAxisAngle([0, 0, -1.], theta) 
        if theta_last is None:
            quat_last = quat
        else:
            quat_last = self.quatFromAxisAngle([0, 0, -1.], theta_last) 

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        # floor 
        states[0, :3] = np.array([x_center, y_curr, 0.])
        states[0, 3:6] = np.array([x_center, y_last, 0.])

        # left wall
        states[1, :3] = np.array([x_center-(dis_x+border)/2., (height+border)/2. + y_curr, 0.])
        states[1, 3:6] = np.array([x_center-(dis_x+border)/2., (height+border)/2. + y_last, 0.])

        # right wall
        states[2, :3] = np.array([x_center+(dis_x+border)/2., (height+border)/2. + y_curr, 0.])
        states[2, 3:6] = np.array([x_center+(dis_x+border)/2., (height+border)/2. + y_last, 0.])

        # back wall
        states[3, :3] = np.array([x_center, (height+border)/2. + y_curr, -(dis_z+border)/2.])
        states[3, 3:6] = np.array([x_center, (height+border)/2. + y_last, -(dis_z+border)/2.])

        # front wall
        states[4, :3] = np.array([x_center, (height+border)/2. + y_curr, (dis_z+border)/2.])
        states[4, 3:6] = np.array([x_center, (height+border)/2. + y_last, (dis_z+border)/2.])

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states


    def set_shape_states(self, glass_states, poured_glass_states):
        '''
        set the the shape states of both glasses.
        '''
        all_states = np.concatenate((glass_states, poured_glass_states), axis = 0)
        pyflex.set_shape_states(all_states)

    def in_glass(self, water, glass_states, border, height):
        '''
        judge whether a water particle is in the poured glass
        water: [x, y, z, 1/m] water particle state.
        '''

        # floor, left, right, back, front
        # state:
        # 0-3: current (x, y, z) coordinate of the center point
        # 3-6: previous (x, y, z) coordinate of the center point
        # 6-10: current quat 
        # 10-14: previous quat 
        x_lower = glass_states[1][0] - border / 2.
        x_upper = glass_states[2][0] + border / 2.
        z_lower = glass_states[3][2] - border / 2.
        z_upper = glass_states[4][2] + border / 2
        y_lower = glass_states[0][1] - border / 2.
        y_upper = glass_states[0][1] + height + border / 2.
        x, y, z = water[:,0], water[:,1], water[:,2]

        res = (x >= x_lower) * (x <= x_upper) * (y >= y_lower) * (y <= y_upper) * (z >= z_lower) * (z <= z_upper)
        res = np.sum(res)
        return res


    # def in_glass2(self, water, glass_states, border, height):
    #     '''
    #     judge whether a water particle is in the poured glass
    #     water: [x, y, z, 1/m] water particle state.
    #     '''

    #     # floor, left, right, back, front
    #     # state:
    #     # 0-3: current (x, y, z) coordinate of the center point
    #     # 3-6: previous (x, y, z) coordinate of the center point
    #     # 6-10: current quat 
    #     # 10-14: previous quat 
    #     x_lower = glass_states[1][0] - border / 2.
    #     x_upper = glass_states[2][0] + border / 2.
    #     z_lower = glass_states[3][2] - border / 2.
    #     z_upper = glass_states[4][2] + border / 2
    #     y_lower = glass_states[0][1] - border / 2.
    #     y_upper = glass_states[0][1] + height + border / 2.
    #     x, y, z = water[0], water[1], water[2]

    #     if x >= x_lower and x <= x_upper and y >= y_lower and y <= y_upper and z >= z_lower and z <= z_upper:
    #         return 1
    #     else:
    #         return 0


    def judge_glass_collide(self, new_states, rotation):
        '''
        judge if the right wall of the pouring glass would collide with the left wall of the poured glass. 
        '''
        right_wall_center = new_states[2][:3]
        pouring_left_wall_center = new_states[1][:3]
        left_wall_center = self.poured_glass_states[1][:3]

        r_corner1_relative_cord = np.array([self.border/2., self.height/2., self.glass_dis_z / 2 + self.border])
        r_corner1_real = rotate_rigid_object(center=right_wall_center, axis=np.array([0, 0, -1]), angle = rotation, relative=r_corner1_relative_cord)
        
        r_corner3_relative_cord = np.array([self.border/2., -self.height/2., -self.glass_dis_z / 2 - self.border])
        r_corner3_real = rotate_rigid_object(center=right_wall_center, axis=np.array([0, 0, -1]), angle = rotation, relative=r_corner3_relative_cord)

        r_corner5_relative_cord = np.array([-self.border/2., -self.height/2., self.glass_dis_z / 2 + self.border])
        r_corner5_real = rotate_rigid_object(center=pouring_left_wall_center, axis=np.array([0, 0, -1]), angle = rotation, relative=r_corner5_relative_cord)

        r_corner8_relative_cord = np.array([-self.border/2., self.height/2., self.glass_dis_z / 2 + self.border])
        r_corner8_real = rotate_rigid_object(center=pouring_left_wall_center, axis=np.array([0, 0, -1]), angle = rotation, relative=r_corner8_relative_cord)

        right_polygon = Polygon([r_corner1_real[:2], r_corner3_real[:2], r_corner5_real[:2], r_corner8_real[:2]])

        leftx, lefty = left_wall_center[0], left_wall_center[1]
        border = self.poured_border
        l_corner1 = np.array([leftx - border / 2, lefty + self.poured_height / 2])
        l_corner2 = np.array([leftx + border / 2, lefty + self.poured_height / 2])
        l_corner3 = np.array([leftx + border / 2, lefty - self.poured_height / 2])
        l_corner4 = np.array([leftx - border / 2, lefty - self.poured_height / 2])
        left_polygon = Polygon([l_corner1, l_corner2, l_corner3, l_corner4])

        res = right_polygon.intersects(left_polygon)

        # rightwalls = [r_corner1_real, r_corner2_real, r_corner3_real, r_corner4_real, r_corner5_real, r_corner6_real, r_corner7_real, r_corner8_real]
        # leftwalls = [l_corner1, l_corner2, l_corner3, l_corner4, l_corner5, l_corner6, l_corner7, l_corner8]
        # if res or self.time_step % 50 == 0:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter([x[0] for x in rightwalls], [x[1] for x in rightwalls], [x[2] for x in rightwalls])
        #     ax.scatter([x[0] for x in leftwalls], [x[1] for x in leftwalls], [x[2] for x in leftwalls])
        #     # ax.scatter(*left_polygon.exterior.xyz)
        #     # ax.scatter(*right_polygon.exterior.xyz)

        #     plt.show()
        #     plt.close()

        return res