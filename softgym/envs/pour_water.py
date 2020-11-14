import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.fluid_env import FluidEnv
import copy
from softgym.utils.misc import rotate_rigid_object, quatFromAxisAngle
from shapely.geometry import Polygon
import random, math


class PourWaterPosControlEnv(FluidEnv):
    def __init__(self, observation_mode, action_mode,
                 config=None, cached_states_path='pour_water_init_states.pkl', **kwargs):
        '''
        This class implements a pouring water task.
        
        observation_mode: "cam_rgb" or "point_cloud" or "key_point"
        action_mode: "rotation_bottom, rotation_top"
        
        '''
        assert observation_mode in ['cam_rgb', 'point_cloud', 'key_point']
        assert action_mode in ['rotation_bottom', 'rotation_top']
        if action_mode == 'rotation_top':
            cached_states_path = 'pour_water_init_states_top.pkl'

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.wall_num = 5  # number of glass walls. floor/left/right/front/back
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        if observation_mode in ['point_cloud', 'key_point']:
            if observation_mode == 'key_point':
                obs_dim = 0
                obs_dim += 13  # Pos (x, z, theta) and shape (w, h, l) of the two cups and the water height.
            else:
                max_particle_num = 13 * 13 * 13 * 4
                obs_dim = max_particle_num * 3
                self.particle_obs_dim = obs_dim
            # z and theta of the second cup (poured_glass) does not change and thus are omitted.
            # add: frac of water in control cup, frac of water in target cup
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        default_config = self.get_default_config()
        border = default_config['glass']['border']
        if action_mode in ["rotation_bottom", "rotation_top"]:
            self.action_direct_dim = 3
            # control the (x, y) corrdinate of the floor center, and theta its rotation angle.
            action_low = np.array([-0.01, -0.01, -0.015])
            action_high = np.array([0.01, 0.01, 0.015])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
        else:
            raise NotImplementedError

        self.prev_reward = 0
        self.reward_min = 0
        self.reward_max = 1
        self.reward_range = self.reward_max - self.reward_min

    def get_default_config(self):
        config = {
            'fluid': {
                'radius': 0.033,
                'rest_dis_coef': 0.55,
                'cohesion': 0.1,  # not actually used, instead, is computed as viscosity * 0.01
                'viscosity': 2,
                'surfaceTension': 0,
                'adhesion': 0.0,  # not actually used, instead, is computed as viscosity * 0.001
                'vorticityConfinement': 40,
                'solidpressure': 0.,
                'dim_x': 8,
                'dim_y': 18,
                'dim_z': 8,
            },
            'glass': {
                'border': 0.045,
                'height': 0.6,
                'glass_distance': 1.0,
                'poured_border': 0.04,
                'poured_height': 0.6,
            },
            'camera_name': 'default_camera',
        }
        return config

    def generate_env_variation(self, num_variations=5, config=None, **kwargs):

        dim_xs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        dim_zs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        self.cached_configs = []
        self.cached_init_states = []
        if config is None:
            config = self.get_default_config()
        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]
        for idx in range(num_variations):
            print("pour water generate env variations {}".format(idx))
            dim_x = random.choice(dim_xs)
            dim_z = random.choice(dim_zs)
            m = min(dim_x, dim_z)
            p = np.random.rand()
            water_radius = config['fluid']['radius'] * config['fluid']['rest_dis_coef']
            if p < 0.5:  # midium water volumes
                print("generate env variation: medium volume water")
                dim_y = int(3.5 * m)
                v = dim_x * dim_y * dim_z
                h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 2
                glass_height = h + (np.random.rand() - 0.5) * 0.001 + config['glass']['border']
            else:
                print("generate env variation: large volume water")
                dim_y = 4 * m
                v = dim_x * dim_y * dim_z
                h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 3
                glass_height = h + (m + np.random.rand()) * 0.001 + config['glass']['border']

            config_variations[idx]['fluid']['dim_x'] = dim_x
            config_variations[idx]['fluid']['dim_y'] = dim_y
            config_variations[idx]['fluid']['dim_z'] = dim_z
            # if you want to change viscosity also, uncomment this
            # config_variations[idx]['fluid']['viscosity'] = self.rand_float(2.0, 10.0)

            config_variations[idx]['glass']['height'] = glass_height
            config_variations[idx]['glass']['poured_height'] = glass_height + np.random.rand() * 0.1
            config_variations[idx]['glass']['glass_distance'] = self.rand_float(0.05 * m, 0.09 * m) + (dim_x + 4) * water_radius / 2.
            config_variations[idx]['glass']['poured_border'] = 0.03

            self.set_scene(config_variations[idx])
            init_state = copy.deepcopy(self.get_state())

            self.cached_configs.append(config_variations[idx])
            self.cached_init_states.append(init_state)

        combined = [self.cached_configs, self.cached_init_states]

        return self.cached_configs, self.cached_init_states

    def get_config(self):
        if self.deterministic:
            config_idx = 0
        else:
            config_idx = np.random.randint(len(self.config_variations))

        self.config = self.config_variations[config_idx]
        return self.config

    def _reset(self):
        '''
        reset to environment to the initial state.
        return the initial observation.
        '''
        self.inner_step = 0
        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        pyflex.step(render=True)
        return self._get_obs()

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'glass_x': self.glass_x, 'glass_y': self.glass_y, 'glass_rotation': self.glass_rotation,
                'glass_states': self.glass_states, 'poured_glass_states': self.poured_glass_states,
                'glass_params': self.glass_params, 'config_id': self.current_config_id}

    def set_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])
        self.glass_x = state_dic['glass_x']
        self.glass_y = state_dic['glass_y']
        self.glass_rotation = state_dic['glass_rotation']
        self.glass_states = state_dic['glass_states']
        self.poured_glass_states = state_dic['poured_glass_states']
        for _ in range(5):
            pyflex.step()

    def initialize_camera(self):
        self.camera_params = {
            'default_camera': {'pos': np.array([1.4, 1.5, 0.1]),
                               'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
                               'width': self.camera_width,
                               'height': self.camera_height},

            'cam_2d': {'pos': np.array([0.5, .7, 4.]),
                       'angle': np.array([0, 0, 0.]),
                       'width': self.camera_width,
                       'height': self.camera_height}
        }

    def set_poured_glass_params(self, config):
        params = config

        self.glass_distance = params['glass_distance']
        self.poured_border = params['poured_border']
        self.poured_height = params['poured_height']

        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        self.poured_glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.07  # glass floor length
        self.poured_glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.07  # glass width

        params['poured_glass_dis_x'] = self.poured_glass_dis_x
        params['poured_glass_dis_z'] = self.poured_glass_dis_z
        params['poured_glass_x_center'] = self.x_center + params['glass_distance']

        self.glass_params.update(params)

    def set_pouring_glass_params(self, config):
        params = config

        self.border = params['border']
        self.height = params['height']

        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.1   # glass floor length
        self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.1  # glass width

        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['glass_x_center'] = self.x_center

        self.glass_params = params

    def set_scene(self, config, states=None, create_only=False):
        '''
        Construct the pouring water scence.
        '''
        # create fluid
        super().set_scene(config)  # do not sample fluid parameters, as it's very likely to generate very strange fluid

        # compute glass params
        if states is None:
            self.set_pouring_glass_params(config["glass"])
            self.set_poured_glass_params(config["glass"])
        else:
            glass_params = states['glass_params']
            self.border = glass_params['border']
            self.height = glass_params['height']
            self.glass_dis_x = glass_params['glass_dis_x']
            self.glass_dis_z = glass_params['glass_dis_z']
            self.glass_distance = glass_params['glass_distance']
            self.poured_border = glass_params['poured_border']
            self.poured_height = glass_params['poured_height']
            self.poured_glass_dis_x = glass_params['poured_glass_dis_x']
            self.poured_glass_dis_z = glass_params['poured_glass_dis_z']
            self.glass_params = glass_params

        # create pouring glass & poured glass
        self.create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)
        self.create_glass(self.poured_glass_dis_x, self.poured_glass_dis_z, self.poured_height, self.poured_border)

        # move pouring glass to be at ground
        self.glass_states = self.init_glass_state(self.x_center, 0, self.glass_dis_x, self.glass_dis_z, self.height, self.border)

        # move poured glass to be at ground
        self.poured_glass_states = self.init_glass_state(self.x_center + self.glass_distance, 0,
                                                         self.poured_glass_dis_x, self.poured_glass_dis_z, self.poured_height, self.poured_border)

        self.set_shape_states(self.glass_states, self.poured_glass_states)

        # record glass floor center x, y, and rotation
        self.glass_x = self.x_center
        if self.action_mode == 'rotation_bottom':
            self.glass_y = 0
        elif self.action_mode == 'rotation_top':
            self.glass_y = 0.5 * self.border + self.height
        self.glass_rotation = 0

        # only create the glass and water, without setting their states
        # this is only used in the pourwater amount env.
        if create_only:
            return

        # no cached init states passed in 
        if states is None:
            fluid_pos = np.ones((self.particle_num, self.dim_position))

            # move water all inside the glass
            fluid_radius = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            fluid_dis = np.array([1.0 * fluid_radius, fluid_radius * 0.5, 1.0 * fluid_radius])
            lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 2. + self.glass_params['border']
            lower_z = -self.glass_params['glass_dis_z'] / 2 + self.glass_params['border']
            lower_y = self.glass_params['border']
            if self.action_mode in ['sawyer', 'franka']:
                lower_y += 0.56  # NOTE: robotics table
            lower = np.array([lower_x, lower_y, lower_z])
            cnt = 0
            rx = int(self.fluid_params['dim_x'] * 1)
            ry = int(self.fluid_params['dim_y'] * 1)
            rz = int(self.fluid_params['dim_z'] / 1)
            for x in range(rx):
                for y in range(ry):
                    for z in range(rz):
                        fluid_pos[cnt][:3] = lower + np.array([x, y, z]) * fluid_dis  # + np.random.rand() * 0.01
                        cnt += 1

            pyflex.set_positions(fluid_pos)
            print("stablize water!")
            for _ in range(100):
                pyflex.step()

            state_dic = self.get_state()
            water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
            in_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
            not_in_glass = 1 - in_glass
            not_total_num = np.sum(not_in_glass)

            while not_total_num > 0:
                max_height_now = np.max(water_state[:, 1])
                fluid_dis = np.array([1.0 * fluid_radius, fluid_radius * 1, 1.0 * fluid_radius])
                lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 4
                lower_z = -self.glass_params['glass_dis_z'] / 4
                lower_y = max_height_now
                lower = np.array([lower_x, lower_y, lower_z])
                cnt = 0
                dim_x = config['fluid']['dim_x']
                dim_z = config['fluid']['dim_z']
                for w_idx in range(len(water_state)):
                    if not in_glass[w_idx]:
                        water_state[w_idx][:3] = lower + fluid_dis * np.array([cnt % dim_x, cnt // (dim_x * dim_z), (cnt // dim_x) % dim_z])
                        cnt += 1

                pyflex.set_positions(water_state)
                for _ in range(40):
                    pyflex.step()

                state_dic = self.get_state()
                water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
                in_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
                not_in_glass = 1 - in_glass
                not_total_num = np.sum(not_in_glass)

            for _ in range(30):
                pyflex.step()
        else:  # set to passed-in cached init states
            self.set_state(states)

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_width, self.camera_height)
        elif self.observation_mode == 'point_cloud':
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
            pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
            pos[:len(particle_pos)] = particle_pos
            return pos.flatten()
        elif 'key_point' in self.observation_mode:
            pos = np.empty(0, dtype=np.float)

            water_state = pyflex.get_positions().reshape([-1, 4])
            in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
            in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
            in_poured_glass = float(np.sum(in_poured_glass)) / len(water_state)
            in_control_glass = float(np.sum(in_control_glass)) / len(water_state)

            cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation, self.glass_dis_x, self.glass_dis_z, self.height,
                                self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                self._get_current_water_height(), in_poured_glass, in_control_glass])

            return np.hstack([pos, cup_state]).flatten()
        
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """
        The reward is computed as the fraction of water in the poured glass.
        NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
        """
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        good_water = in_poured_glass * (1 - in_control_glass)
        good_water_num = np.sum(good_water)

        reward = float(good_water_num) / water_num
        return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        good_water = in_poured_glass * (1 - in_control_glass)
        good_water_num = np.sum(good_water)

        performance = float(good_water_num) / water_num
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance

        return {
            'normalized_performance': (performance - performance_init) / (self.reward_max - performance_init),
            'performance': performance
        }

    def _step(self, action):
        '''
        action: np.ndarray of dim 1x3, (x, y, theta). (x, y) specifies the floor center coordinate, and theta 
            specifies the rotation.
        '''
        # make action as increasement, clip its range
        move = action[:2]
        rotate = action[2]
        move = np.clip(move, a_min=self.action_space.low[0], a_max=self.action_space.high[0])
        rotate = np.clip(rotate, a_min=self.action_space.low[2], a_max=self.action_space.high[2])
        dx, dy, dtheta = move[0], move[1], rotate
        x, y, theta = self.glass_x + dx, self.glass_y + dy, self.glass_rotation + dtheta

        # check if the movement of the pouring glass collide with the poured glass.
        # the action only take effects if there is no collision
        new_states = self.rotate_glass(self.glass_states, x, y, theta)
        if not self.judge_glass_collide(new_states, theta) and self.above_floor(new_states, theta):
            self.glass_states = new_states
            self.glass_x, self.glass_y, self.glass_rotation = x, y, theta
        else:  # invalid move, old state becomes the same as the current state
            self.glass_states[:, 3:6] = self.glass_states[:, :3].copy()
            self.glass_states[:, 10:] = self.glass_states[:, 6:10].copy()

        # pyflex takes a step to update the glass and the water fluid
        self.set_shape_states(self.glass_states, self.poured_glass_states)
        pyflex.step(render=True)

        self.inner_step += 1

    def create_glass(self, glass_dis_x, glass_dis_z, height, border):
        """
        the glass is a box, with each wall of it being a very thin box in Flex.
        each wall of the real box is represented by a box object in Flex with really small thickness (determined by the param border)
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
        halfEdge = np.array([glass_dis_x / 2. + border, border / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # left wall
        halfEdge = np.array([border / 2., (height) / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # right wall
        boxes.append([halfEdge, center, quat])

        # back wall
        halfEdge = np.array([(glass_dis_x) / 2., (height) / 2., border / 2.])
        boxes.append([halfEdge, center, quat])

        # front wall
        boxes.append([halfEdge, center, quat])

        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)

        return boxes

    def rotate_glass(self, prev_states, x, y, theta):
        '''
        given the previous states of the glass, rotate it with angle theta.
        update the states of the 5 boxes that form the box: floor, left/right wall, back/front wall. 
        rotate the glass, where the center point is the center of the floor or the top.
        
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

        if self.action_mode == 'rotation_bottom':
            # floor: center position does not change
            states[0, :3] = np.array([x_center, y, 0.])

            # left wall: center must move right and move down. 
            relative_coord = np.array([-(dis_x+ border) / 2., (self.height) / 2., 0.])
            states[1, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # right wall
            relative_coord = np.array([(dis_x+ border) / 2., (self.height) / 2., 0.])
            states[2, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # back wall
            relative_coord = np.array([0, (self.height) / 2., -(dis_z+ border) / 2.])
            states[3, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # front wall
            relative_coord = np.array([0, (self.height) / 2., (dis_z+ border) / 2.])
            states[4, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

        elif self.action_mode == 'rotation_top':
            # floor
            relative_coord = np.array([0, -self.height, 0.])
            states[0, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # left wall
            relative_coord = np.array([-(dis_x+ border) / 2., -self.height / 2., 0.])
            states[1, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # right wall
            relative_coord = np.array([(dis_x+ border) / 2., -self.height / 2., 0.])
            states[2, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # back wall
            relative_coord = np.array([0, -self.height / 2., -(dis_z+ border) / 2.])
            states[3, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

            # front wall
            relative_coord = np.array([0, -self.height / 2., (dis_z) / 2.])
            states[4, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta, relative=relative_coord)

        states[:, 6:10] = quat_curr

        return states

    def init_glass_state(self, x, y, glass_dis_x, glass_dis_z, height, border):
        '''
        set the initial state of the glass.
        '''
        dis_x, dis_z = glass_dis_x, glass_dis_z
        x_center, y_curr, y_last = x, y, 0.
        quat = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        # floor 
        states[0, :3] = np.array([x_center, y_curr, 0.])
        states[0, 3:6] = np.array([x_center, y_last, 0.])

        # left wall
        states[1, :3] = np.array([x_center - (dis_x  + border) / 2., (height ) / 2. + y_curr, 0.])
        states[1, 3:6] = np.array([x_center - (dis_x + border) / 2., (height ) / 2. + y_last, 0.])

        # right wall
        states[2, :3] = np.array([x_center + (dis_x + border) / 2., (height ) / 2. + y_curr, 0.])
        states[2, 3:6] = np.array([x_center + (dis_x + border) / 2., (height ) / 2. + y_last, 0.])

        # back wall
        states[3, :3] = np.array([x_center, (height ) / 2. + y_curr, -(dis_z + border) / 2.])
        states[3, 3:6] = np.array([x_center, (height ) / 2. + y_last, -(dis_z + border) / 2.])

        # front wall
        states[4, :3] = np.array([x_center, (height ) / 2. + y_curr, (dis_z + border) / 2.])
        states[4, 3:6] = np.array([x_center, (height ) / 2. + y_last, (dis_z + border) / 2.])

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states

    def set_shape_states(self, glass_states, poured_glass_states):
        '''
        set the the shape states of both glasses.
        '''
        all_states = np.concatenate((glass_states, poured_glass_states), axis=0)
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
        x, y, z = water[:, 0], water[:, 1], water[:, 2]

        res = (x >= x_lower) * (x <= x_upper) * (y >= y_lower) * (y <= y_upper) * (z >= z_lower) * (z <= z_upper)
        return res

    def judge_glass_collide(self, new_states, rotation):
        '''
        judge if the front wall of the pouring glass would collide with the front wall of the poured glass. 
        '''
        pouring_right_wall_center = new_states[2][:3]
        pouring_left_wall_center = new_states[1][:3]

        # build the corner of the front wall of the control glass
        r_corner1_relative_cord = np.array([self.border / 2., self.height / 2., self.glass_dis_z / 2 + self.border])
        r_corner1_real = rotate_rigid_object(center=pouring_right_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                             relative=r_corner1_relative_cord)

        r_corner3_relative_cord = np.array([self.border / 2., -self.height / 2., self.glass_dis_z / 2 - self.border])
        r_corner3_real = rotate_rigid_object(center=pouring_right_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                             relative=r_corner3_relative_cord)

        r_corner5_relative_cord = np.array([-self.border / 2., -self.height / 2., self.glass_dis_z / 2 + self.border])
        r_corner5_real = rotate_rigid_object(center=pouring_left_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                             relative=r_corner5_relative_cord)

        r_corner8_relative_cord = np.array([-self.border / 2., self.height / 2., self.glass_dis_z / 2 + self.border])
        r_corner8_real = rotate_rigid_object(center=pouring_left_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                             relative=r_corner8_relative_cord)

        control_polygon = Polygon([r_corner1_real[:2], r_corner3_real[:2], r_corner5_real[:2], r_corner8_real[:2]])

        left_wall_center = self.poured_glass_states[1][:3]
        leftx, lefty = left_wall_center[0], left_wall_center[1]
        right_wall_center = self.poured_glass_states[2][:3]
        rightx, righty = right_wall_center[0], right_wall_center[1]
        border = self.poured_border
        target_front_corner1 = np.array([leftx - border / 2, lefty + self.poured_height / 2])
        traget_front_corner2 = np.array([leftx - border / 2, lefty - self.poured_height / 2])
        traget_front_corner3 = np.array([rightx + border / 2, righty - self.poured_height / 2])
        target_front_corner4 = np.array([rightx + border / 2, righty + self.poured_height / 2])
        target_polygon = Polygon([target_front_corner1, traget_front_corner2, traget_front_corner3, target_front_corner4])

        res = control_polygon.intersects(target_polygon)

        return res

    def above_floor(self, states, rotation):
        '''
        judge all the floors are above the ground.
        '''

        floor_center = states[0][:3]
        corner_relative = [
            np.array([self.glass_dis_x / 2., -self.border / 2., self.glass_dis_z / 2.]),
            np.array([self.glass_dis_x / 2., -self.border / 2., -self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., -self.border / 2., self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., -self.border / 2., -self.glass_dis_z / 2.]),

            np.array([self.glass_dis_x / 2., self.border / 2. + self.height, self.glass_dis_z / 2.]),
            np.array([self.glass_dis_x / 2., self.border / 2. + self.height, -self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., self.border / 2. + self.height, self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., self.border / 2. + self.height, -self.glass_dis_z / 2.]),
        ]

        for corner_rel in corner_relative:
            corner_real = rotate_rigid_object(center=floor_center, axis=np.array([0, 0, -1]), angle=rotation,
                                              relative=corner_rel)
            if corner_real[1] < - self.border / 2:
                return False

        return True