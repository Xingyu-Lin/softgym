import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.fluid_env import FluidEnv
from softgym.utils.misc import rotate_rigid_object, quatFromAxisAngle


class PourJamPosControlEnv(FluidEnv):
    def __init__(self, observation_mode, action_mode, horizon = 400, deterministic = False, render_mode = 'particle'):
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

        super().__init__(horizon, deterministic, render_mode)
        assert observation_mode in ['cam_rgb', 'full_state']
        assert action_mode in ['direct'] 

        if observation_mode == 'cam_rgb':
            self.observation_space = Box(low = -np.inf, high = np.inf, shape = (self.camera_height, self.camera_width, 3), 
                dtype=np.float32)
        else:
            raise NotImplementedError

        if action_mode == 'direct':
            self.action_direct_dim = 3 # control the (x, y) corrdinate of the floor center, and theta its rotation angel.
            self.action_space = Box(np.array([-1.] * self.action_direct_dim), np.array([1.] * self.action_direct_dim), dtype=np.float32)
        else:
            raise NotImplementedError

        self.init_flex_state = self.get_state()
        

    def _reset(self):
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

    def set_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])

    def initialize_camera(self):
        '''
        set the camera width, height, position and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        x_center = self.x_center # center of the glass floor
        z = self.fluid_params['z'] # lower corner of the water fluid along z-axis.
        self.camera_params = {
                        'pos': np.array([x_center + 0.5, 0.4 + 2.5, z + 0.8]),
                        'angle': np.array([0., -70/180. * np.pi, 0.]),
                        'width': self.camera_width,
                        'height': self.camera_height
                        }

    def sample_glass_params(self):
        params = {}
        params['border_range'] = 0.015, 0.025
        params['height_range'] = 0.5, 0.7
        params['plate_distance_range'] = 0.5, 0.8
        params['plate_height_range'] = 0.1, 0.2

        # make pouring glass params
        
        if not self.deterministic: 
            self.border = self.rand_float(params['border_range'][0], params['border_range'][1]) # the thickness of the glass wall.
            self.height = self.rand_float(params['height_range'][0], params['height_range'][1]) # the height of the glass
            fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + self.rand_float(0., 0.1) # glass floor length
            self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + self.rand_float(0, 0.1) # glass width
        else:
            self.border = 0.02
            self.height = 0.4
            fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.05 # glass floor length
            self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.05 # glass width
        
        params['border'] = self.border
        params['height'] = self.height
        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['init_glass_x_center'] = self.x_center

        # make jam plate params
        self.plate_num = 2
        self.plate_distances = np.zeros(self.plate_num )
        self.plate_centers = np.zeros(self.plate_num )
        self.plate_borders = np.zeros(self.plate_num )
        self.plate_heights = np.zeros(self.plate_num )
        self.plate_dis_x = np.zeros(self.plate_num )
        self.plate_dis_z = np.zeros(self.plate_num)
        for i in range(self.plate_num):
            if not self.deterministic:
                self.plate_distances[i] = self.rand_float(params['plate_distance_range'][0], params['plate_distance_range'][1]) # distance between the pouring glass and the poured glass
                self.plate_borders[i] = self.rand_float(params['border_range'][0], params['border_range'][1])
                self.plate_heights[i] = self.rand_float(params['plate_height_range'][0], params['plate_height_range'][1])
                self.plate_dis_x[i] = (self.fluid_params['dim_x'] * fluid_radis + self.rand_float(0., 0.2)) * 0.7
                self.plate_dis_z[i] = (self.fluid_params['dim_z'] * fluid_radis + self.rand_float(0., 0.2)) * 0.7
            else:
                self.plate_distances[i] = 0.6
                self.plate_borders[i] = 0.01
                self.plate_heights[i] = 0.15
                self.plate_dis_x[i] = (self.fluid_params['dim_x'] * fluid_radis + 0.2) * 0.7 
                self.plate_dis_z[i] = (self.fluid_params['dim_z'] * fluid_radis + 0.2) * 0.7
            
            self.plate_centers[i] = self.x_center
            for j in range(i + 1):
                self.plate_centers[i] += self.plate_distances[j]

        params['plate_center'] = self.plate_centers
        params['plate_border'] = self.plate_borders
        params['plate_height'] = self.plate_heights
        params['plate_dis_x'] = self.plate_dis_x
        params['plate_dix_z'] = self.plate_dis_z

        self.glass_params = params

    def deterministic_fluid_params(self):
        '''
        sample params for the fluid.
        '''
        params = {}
        params['radius'] = 0.1
        params['rest_dis_coef'] = 0.45
        params['cohesion'] = 0.018
        params['viscosity'] = 2.0
        params['surfaceTension'] = 0.0
        params['adhesion'] = 0.0
        params['vorticityConfinement'] = 40.0
        params['solidpressure'] = 0.

        params['dim_x'] = 3
        params['dim_y'] = 15
        params['dim_z'] = 3

        return params

    def set_scene(self):
        '''
        Construct the pouring water scence.
        '''

        # create fluid 
        if self.deterministic:
            super().set_scene(self.deterministic_fluid_params())
        else:
            super().set_scene()

        # compute glass params
        self.sample_glass_params()

        # create pouring glass
        glass = self.create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)
        for i in range(len(glass)):
            halfEdge = glass[i][0]
            center = glass[i][1]
            quat = glass[i][2]
            pyflex.add_box(halfEdge, center, quat)

        # create poured jam plate
        for j in range(self.plate_num):
            plate = self.create_glass(self.plate_dis_x[j], self.plate_dis_z[j], self.plate_heights[j], self.plate_borders[j])
            for i in range(len(plate)):
                halfEdge = plate[i][0]
                center = plate[i][1]
                quat = plate[i][2]
                pyflex.add_box(halfEdge, center, quat)
        
        # move pouring glass to be at ground
        self.glass_states = self.init_glass_state(self.x_center, 0, self.glass_dis_x, self.glass_dis_z, self.height, self.border)

        # move poured glass to be at ground
        self.plate_states = np.zeros((self.plate_num * self.wall_num, self.dim_shape_state))
        for j in range(self.plate_num):
            self.plate_states[j*self.wall_num:(j+1)*self.wall_num] = self.init_glass_state(self.plate_centers[j], 0, 
                self.plate_dis_x[j], self.plate_dis_z[j], self.plate_heights[j], self.plate_borders[j])

        self.set_shape_states(self.glass_states, self.plate_states)

        # give some time for water to stablize 
        for i in range(100):
            pyflex.step()

        # record glass floor center x, y, and rotation
        self.action_x = self.x_center
        self.action_y = 0
        self.rotation = 0

        # print("pour water inital scene constructed over...")

    def get_current_observation(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_rgb':
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
        in_glass = 0
        for water in water_state:
            res = self.in_glass(water)
            in_glass += res
            
        if self.debug:
            print("water num: ", water_num, "in glass num: ", in_glass)
        return float(in_glass) / water_num


    def _step(self, action):
        '''
        action: np.ndarray of dim 1x3, (x, y, theta). (x, y) specifies the floor center coordinate, and theta 
            specifies the rotation.
        
        return: gym-like (next_obs, reward, done, info)
        '''

        # make action as increasement
        # move = action[:2]
        # rotate = action[2]
        # move = np.clip(move, a_min = -0.3, a_max = 0.3)
        # rotate = np.clip(rotate, a_min = -0.2, a_max = 0.2)
        # x, y, theta = move[0], move[1], rotate
        # x, y, theta = self.action_x + x, self.action_y + y, self.rotation + theta
        # self.action_x, self.action_y, self.rotation = x, y, theta

        # this directly sets the glass x,y and rotation
        x, y, theta = action[0], action[1], action[2]

        self.glass_states = self.rotate_glass(self.glass_states, x, y, theta)

        # pyflex takes a step to update the glass and the water fluid
        self.set_shape_states(self.glass_states, self.plate_states)
        if self.record_video:
            pyflex.step(capture = 1, path = self.video_path + 'render_' + str(self.time_step) + '.tga')
        else:
            pyflex.step() 

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
        quat = quatFromAxisAngle([0, 0, -1.], 0.) 
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
        quat_curr = quatFromAxisAngle([0, 0, -1.], theta) 

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
        quat = quatFromAxisAngle([0, 0, -1.], 0.) 

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

    def in_glass(self, water):
        '''
        judge whether a water particle is in the poured glass
        water: [x, y, z, 1/m] water particle state.
        '''
        return 0
        # floor, left, right, back, front
        # state:
        # 0-3: current (x, y, z) coordinate of the center point
        # 3-6: previous (x, y, z) coordinate of the center point
        # 6-10: current quat 
        # 10-14: previous quat 
        # x_lower = self.poured_glass_states[1][0]
        # x_upper = self.poured_glass_states[2][0]
        # z_lower = self.poured_glass_states[3][2]
        # z_upper = self.poured_glass_states[4][2]
        # y_lower = self.poured_border
        # y_upper = self.poured_height
        # x, y, z = water[0], water[1], water[2]
        # if x >= x_lower and x <= x_upper and y >= y_lower and y <= y_upper and z >= z_lower and z <= z_upper:
        #     return 1
        # else:
        #     return 0



