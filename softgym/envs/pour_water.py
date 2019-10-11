import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.flex_env import FlexEnv
import time
import copy


class PourWaterPosControlEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode):
        '''
        This class implements a pouring water task.
        TODO: add more description of the task.
        TODO: allow parameter configuring of the scence.
        '''
       
        self.dim_x = 5 # num of grid points in x-axis for the water fluid.
        self.dim_y = 20 
        self.dim_z = 3
        self.border = 0.025 # the thickness of the glass wall.
        self.height = 0.5 # the height of the glass.

        self.observation_mode = observation_mode
        self.action_mode = action_mode

        self.dim_shape_state = 14 # dimension of a shape object in Flex
        self.dim_position = 4
        self.dim_velocity = 3
        self.wall_num = 5 # number of glass walls. floor/left/right/front/back 

        super().__init__()
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb'] 
        assert action_mode in ['direct'] 

        if observation_mode == 'key_point':
            # obs is the center point of each wall
            # TODO: figure out what observation is
            self.observation_space = Box(np.array([-np.inf] * self.wall_num), np.array([np.inf] * self.wall_num), dtype=np.float32)
            # self.obs_key_point_idx = self.get_obs_key_point_idx()
        else:
            raise NotImplementedError

        if action_mode == 'direct':
            self.action_direct_dim = 3 # control the (x, y) corrdinate of the floor center, and theta its rotation angel.
            self.action_space = Box(np.array([-1.] * self.action_direct_dim), np.array([1.] * self.action_direct_dim), dtype=np.float32)
            # self.action_key_point_idx = self.get_action_key_point_idx()
        else:
            raise NotImplementedError

        self.init_flex_state = self.get_state()
        

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
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position}

    # def set_flex_state(self, state_dic):
    #     '''
    #     set the postion, velocity of flex particles, and postions of flex shapes.
    #     '''
    #     pyflex.set_positions(state_dic["particle_pos"])
    #     pyflex.set_velocities(state_dic["particle_vel"])
    #     pyflex.set_shape_states(state_dic["shape_pos"])

    def set_scene(self):
        '''
        Construct the pouring water scence.
        '''
        
        # create water fluid
        x_center = self.rand_float(-0.2, 0.2) # center of the glass floor
        x = x_center - (self.dim_x-1)/2.*0.055 # lower corner of the water fluid grid along x-axis. 0.055 is the grid step size, which is hard-coded in the cpp file.
        y = 0.055/2. + self.border + 0.01 # lower corner of the water fluid along y-axis.
        z = 0. - (self.dim_z-1)/2.*0.055 # lower corner of the water fluid along z-axis.
        self.glass_dis_x = self.dim_x * 0.055 + self.rand_float(0., 0.3) # glass floor length
        self.glass_dis_z = 0.2 # glass width
        scene_params = np.array([x, y, z, self.dim_x, self.dim_y, self.dim_z, self.glass_dis_x, self.glass_dis_z])
        pyflex.set_scene(6, scene_params, 0)

        # create glass
        glass = self.create_glass()
        for i in range(len(glass)):
            halfEdge = glass[i][0]
            center = glass[i][1]
            quat = glass[i][2]
            pyflex.add_box(halfEdge, center, quat)
      
        
        # move glass to be at ground
        self.glass_floor_centerx = x_center
        self.glass_floor_centery = 0.
        self.glass_states = self.init_glass_state(self.glass_floor_centerx, self.glass_floor_centery)
        pyflex.set_shape_states(self.glass_states)
        # pyflex.step()

        # give some time for water to stablize 
        # for i in range(20):
        #     pyflex.step()

        print("pour water inital scene constructed over...")

    def get_current_observation(self):
        '''
        return the observation based on the current flex state.
        TODO: figure out the state of the agent.
        '''
        return np.zeros(self.wall_num)

    def compute_reward(self):
        '''
        return the reward for the agent based on his action and the current flex state.
        TODO: figure out the reward of the agent.
        '''
        return 0

    def step(self, action):
        '''
        action: np.ndarray of dim 1x3, (x, y, theta). (x, y) specifies the floor center coordinate, and theta 
            specifies the rotation.
        
        return: gym-like (next_obs, reward, done, info)
        '''
        x, y, theta = action[0], action[1], action[2]
        
        # move and rotate the glass. TODO: check on this
        # after_move_states = self.move_glass(self.glass_states, x, y)
        # after_rotate_states = self.rotate_glass(after_move_states, theta)
        # self.glass_states = after_rotate_states

        if theta == 0:
            # print("just move")
            self.glass_states = self.move_glass(self.glass_states, x, y)
        else:
            self.glass_states = self.rotate_glass(self.glass_states, theta)

        # pyflex takes a step to update the glass and the water fluid
        pyflex.set_shape_states(self.glass_states)
        pyflex.step()

        # get reward and new observation for the agent.
        obs = self.get_current_observation()
        reward = self.compute_reward()

        return obs, reward, False, {}

    def create_glass(self):
        """
        the glass is a box, with each wall of it being a very thin box in Flex.
        each wall of the real box is represented by a box object in FLex with really small thickness (determined by the param border)
        dis_x: the length of the glass
        dis_z: the width of the glass
        height: the height of the glass.
        border: the thickness of the glass wall.

        the halfEdge determines the center point of each wall.
        Note: this is merely setting the length of each dimension of the wall, but not the actual position of them.
        That's why leaf and right walls have exactly the same params, and so do front and back walls.   
        """
        center = np.array([0., 0., 0.])
        quat = np.array([1., 0., 0., 0.])
        boxes = []

        # floor
        halfEdge = np.array([self.glass_dis_x/2., self.border/2., self.glass_dis_z/2.])
        boxes.append([halfEdge, center, quat])

        # left wall
        halfEdge = np.array([self.border/2., (self.height+self.border)/2., self.glass_dis_z/2.])
        boxes.append([halfEdge, center, quat])

        # right wall
        boxes.append([halfEdge, center, quat])

        # back wall
        halfEdge = np.array([(self.glass_dis_x+self.border*2)/2., (self.height+self.border)/2., self.border/2.])
        boxes.append([halfEdge, center, quat])

        # front wall
        boxes.append([halfEdge, center, quat])

        return boxes

    def rotate_glass(self, prev_states, theta):
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

        w = dis_x / 2.
        h = self.height / 2.
        b = self.border / 2.
        border = self.border

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        for i in range(5):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]

        x_center = prev_states[0][0]  # previous x of floor center
        y = prev_states[0][1] # previous y of floor center

        # floor: center position does not change
        states[0, :3] = np.array([x_center, y, 0.])

        # left wall: center must move right and move down. 
        states[1, :3] = np.array([x_center-(w+b)*np.cos(theta) + (h+b)*np.sin(theta),
            y + (w+b)*np.sin(theta) + (h+b)*np.cos(theta), 0.])

        # right wall
        states[2, :3] = np.array([x_center+ (w + b)*np.cos(theta) + (h+b)*np.sin(theta) ,
            y - (w+b)*np.sin(theta) + (h+b)*np.cos(theta), 0.])

        # back wall
        states[3, :3] = np.array([x_center + (h + b)*np.sin(theta), y + (h+b)*np.cos(theta), -(dis_z+border)/2.])

        # front wall
        states[4, :3] = np.array([x_center + (h + b)*np.sin(theta), y + (h+b)*np.cos(theta), (dis_z+border)/2.])

        states[:, 6:10] = quat_curr

        return states

    def move_glass(self, prev_states, x_curr, y_curr):
        '''
        given the prev_states, and current (x, y) of the floor center
        update the states of the 5 shapes that form the glass: floor, left/right wall, back/front wall. 
        
        state:
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat 
        10-14: previous quat 
        '''
        dis_x, dis_z = self.glass_dis_x, self.glass_dis_z
        border, height = self.border, self.height
        quat = self.quatFromAxisAngle([0, 0, -1.], 0.) 

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        for i in range(5): # TODO: check this
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]
            states[i][6:10] = prev_states[i][6:10]

        # floor 
        states[0, :3] = np.array([x_curr, y_curr, 0.])

        # left wall
        states[1, :3] = np.array([x_curr-(dis_x + border)/2., (height + border)/2. + y_curr, 0.])

        # right wall
        states[2, :3] = np.array([x_curr+(dis_x+border)/2., (height+border)/2. + y_curr, 0.])

        # back wall
        states[3, :3] = np.array([x_curr, (height+border)/2. + y_curr, -(dis_z+border)/2.])

        # front wall
        states[4, :3] = np.array([x_curr, (height+border)/2. + y_curr, (dis_z+border)/2.])

        return states

    def init_glass_state(self, x, y):
        '''
        set the initial state of the glass.
        '''
        dis_x, dis_z = self.glass_dis_x, self.glass_dis_z
        x_center, y_curr, y_last  = x, y, 0.05
        height, border = self.height, self.border
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


    def rand_float(self, lo, hi):
        return np.random.rand() * (hi - lo) + lo

    def rand_int(self, lo, hi):
        return np.random.randint(lo, hi)


    def quatFromAxisAngle(self, axis, angle):
        '''
        given a rotation axis and angle, return a quatirian that represents such roatation.
        '''
        axis /= np.linalg.norm(axis)

        half = angle * 0.5
        w = np.cos(half)

        sin_theta_over_two = np.sin(half)
        axis *= sin_theta_over_two

        quat = np.array([axis[0], axis[1], axis[2], w])

        return quat

    # def get_obs_key_point_idx(self):
    #     idx_p1 = 0
    #     idx_p2 = 64 * 31
    #     return np.array([idx_p1, idx_p2])

    # def get_action_key_point_idx(self):
    #     idx_p1 = 0
    #     idx_p2 = 64 * 31
    #     return np.array([idx_p1, idx_p2])
