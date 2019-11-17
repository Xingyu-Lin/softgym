import numpy as np
from gym.spaces import Box

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

class PourWaterPosControlEnv(FluidEnv):
    def __init__(self, observation_mode, action_mode, horizon = 300, deterministic = False, render_mode = 'particle'):
        '''
        This class implements a pouring water task.
        
        observation_mode: "cam_img" or "full_state"
        action_mode: "direct"
        horizon: environment horizon
        
        TODO: add more description of the task.
        TODO: allow parameter configuring of the scence.
        '''

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.wall_num = 5 # number of glass walls. floor/left/right/front/back 

        super().__init__(horizon, deterministic, render_mode)
        assert observation_mode in ['cam_img', 'full_state'] 
        assert action_mode in ['direct'] 

        if observation_mode == 'cam_img':
            self.observation_space = Box(low = -np.inf, high = np.inf, shape = (self.camera_height, self.camera_width, 3), 
                dtype=np.float32)
        else:
            raise NotImplementedError

        if action_mode == 'direct':
            self.action_direct_dim = 3 # control the (x, y) corrdinate of the floor center, and theta its rotation angel.
            self.action_space = Box(np.array([-1.] * self.action_direct_dim), np.array([1.] * self.action_direct_dim), dtype=np.float32)
        else:
            raise NotImplementedError
        
    def reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        self.set_state(self.init_flex_state)
        return self.get_current_observation()
    
    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions().reshape(-1, self.dim_position)
        particle_vel = pyflex.get_velocities().reshape(-1, self.dim_velocity)
        shape_position = pyflex.get_shape_states().reshape(-1, self.dim_shape_state)
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
            'glass_x': self.glass_x, 'glass_y': self.glass_y, 'glass_rotation': self.glass_rotation, 'glass_states': self.glass_states}

    def set_state(self, state_dic):
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

    def initialize_camera(self):
        '''
        set the camera width, height, position and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        x_center = self.x_center # center of the glass floor
        z = self.fluid_params['z'] # lower corner of the water fluid along z-axis.
        self.camera_params = {
                        'pos': np.array([x_center + 1.3, 0.8 + 1.5, z + 0.5]),
                        'angle': np.array([0.4 * np.pi, -70/180. * np.pi, 0]),
                        # 'pos': np.array([x_center -1.3, 0.8, z + 0.5]),
                        # 'angle': np.array([0, 0, -0.5 * np.pi]),
                        'width': self.camera_width,
                        'height': self.camera_height
                        }

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
            self.poured_glass_dis_x = self.fluid_params['dim_x'] * fluid_radis # glass floor length
            self.poured_glass_dis_z = self.fluid_params['dim_z'] * fluid_radis # glass width

        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['poured_glass_dis_x'] = self.poured_glass_dis_x
        params['poured_glass_dis_z'] = self.poured_glass_dis_z
        params['init_glass_x_center'] = self.x_center
        params['poured_glass_x_center'] = self.x_center + params['glass_distance']

        self.glass_params = params

    def set_scene(self):
        '''
        Construct the pouring water scence.
        '''
        # create fluid
        config = open("../softgym/envs/PourWaterDefaultConfig.yaml", 'r')
        config = yaml.load(config)
        if self.deterministic:
            super().set_scene(config["fluid"])
            # super().set_scene(self.deterministic_fluid_params())
        else:
            super().set_scene()

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
        # pyflex.set_shape_states(self.glass_states)

        # give some time for water to stablize 
        for i in range(150):
            pyflex.step()

        # record glass floor center x, y, and rotation
        self.glass_x = self.x_center
        self.glass_y = 0
        self.glass_rotation = 0

        self.init_flex_state = self.get_state()

        print("pour water inital scene constructed over...")

    def get_current_observation(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_img':
            img = pyflex.render()
            width, height = self.camera_width, self.camera_height
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
            return img
        else:
            raise NotImplementedError

    def compute_reward(self):
        '''
        the reward is computed as the fraction of water in the poured glass.
        TODO: do we want to consider the increase of the fraction?
        '''
        state_dic = self.get_state()
        water_state = state_dic['particle_pos']
        water_num = len(water_state)
        
        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)

        # in_poured_glass2 = 0
        # for water in water_state:
        #     res = self.in_glass2(water, self.poured_glass_states, self.poured_border, self.poured_height)
        #     in_poured_glass2 += res

        # assert in_poured_glass == in_poured_glass2

        in_pouring_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)

        # in_pouring_glass2 = 0
        # for water in water_state:
        #     res = self.in_glass2(water, self.glass_states, self.border, self.height)
        #     in_pouring_glass2 += res

        # print(in_pouring_glass, in_pouring_glass2)
        # assert in_pouring_glass == in_pouring_glass2
            
        if self.debug:
            print("water num: ", water_num, "in glass num: ", in_poured_glass)
        return float(in_poured_glass) / water_num #+ 0.1 * float(in_pouring_glass) / water_num

    def compute_in_pouring_glass_water(self):
        state_dic = self.get_state()
        water_state = state_dic['particle_pos']
        in_pouring_glass = 0
        for water in water_state:
            res = self.in_glass(water, self.glass_states, self.border, self.height)
            in_pouring_glass += res
        return in_pouring_glass


    def step(self, action):
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

        # this directly sets the glass x,y and rotation
        # x, y, theta = action[0], action[1], action[2]

        # check if the movement of the pouring glass collide with the poured glass.
        new_states = self.rotate_glass(self.glass_states, x, y, theta)
        if not self.judge_glass_collide(new_states, theta):
            self.glass_states = new_states
            self.glass_x, self.glass_y, self.glass_rotation = x, y, theta
        else:
            print("shapes collide!")

        # pyflex takes a step to update the glass and the water fluid
        self.set_shape_states(self.glass_states, self.poured_glass_states)
        # pyflex.set_shape_states(self.glass_states)
        if self.record_video:
            pyflex.step(capture = 1, path = self.video_path + 'render_' + str(self.time_step) + '.tga')
        else:
            pyflex.step() 

        flex_states = self.get_state()
        new_poured_glass_states = flex_states['shape_pos'][-5:]
        if (new_poured_glass_states == self.poured_glass_states).all():
            pass 
        else: 
            print(self.time_step, " shapes collide!")
        self.poured_glass_states = new_poured_glass_states

        # get reward and new observation for the agent.
        obs = self.get_current_observation()
        reward = self.compute_reward()

        self.time_step += 1

        done = True if self.time_step == self.horizon else False
        return obs, reward, done, {}

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

    def init_glass_state(self, x, y, glass_dis_x, glass_dis_z, height, border):
        '''
        set the initial state of the glass.
        '''
        dis_x, dis_z = glass_dis_x, glass_dis_z
        x_center, y_curr, y_last  = x, y, 0.
        quat = self.quatFromAxisAngle([0, 0, -1.], 0.) 

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