import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.fluid_env import FluidEnv
import time
import copy
import os
from softgym.envs.util import quatFromAxisAngle
from pyquaternion import Quaternion
import random
import yaml, pickle
import os.path as osp


class PassWater1DEnv(FluidEnv):
    def __init__(self, observation_mode, action_mode, config=None, cached_states_path='pass_water_init_states.pkl', **kwargs):
        '''
        This class implements a pouring water task.
        
        observation_mode: "cam_rgb" or "full_state"
        action_mode: "direct"
        
        TODO: add more description of the task.
        '''
        assert observation_mode in ['cam_rgb', 'point_cloud', 'key_point']
        assert action_mode in ['direct']

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.wall_num = 5  # number of glass walls. floor/left/right/front/back
        self.inner_step = 0  # count action repetation
        self.distance_coef = 1.
        self.water_penalty_coef = 10.
        self.terminal_x = 1.2
        self.min_x = -0.5
        self.max_x = 1.5

        super().__init__(**kwargs)

        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_states_path = osp.join(cur_dir, cached_states_path)
        else:
            self.cached_states_path = cached_states_path

        if not self.use_cached_states or self.get_cached_configs_and_states(cached_states_path) is False:
            if config is None:
                config = self.get_default_config()
            self.generate_env_variation(config, num_variations=self.num_variations, save_to_file=True)

        if observation_mode in ['point_cloud', 'key_point']:
            # TODO: change this to this task
            if observation_mode == 'key_point':
                obs_dim = 0
            else:
                obs_dim = 1225 * 3
                self.particle_obs_dim = obs_dim
            # z and theta of the second cup (poured_glass) does not change and thus are omitted.
            obs_dim += 10  # Pos (x, z, theta) and shape (w, h, l) resetof the two cups.
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        if action_mode == 'direct':
            self.action_direct_dim = 1
            # control the (x, y) corrdinate of the floor center, and theta its rotation angle.
            action_low = np.array([-0.025])
            action_high = np.array([0.025])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
        else:
            raise NotImplementedError

    def get_default_config(self):
        config = {
            'fluid': {
                'radius': 0.1,
                'rest_dis_coef': 0.55,
                'cohesion': 0.02,
                'viscosity': 2.0,
                'surfaceTension': 0.,
                'adhesion': 0.0,
                'vorticityConfinement': 40,
                'solidpressure': 0.,
                'dim_x': 8,
                'dim_y': 18,
                'dim_z': 8,
            },
            'glass': {
                'border': 0.025,
                'height': 0.6,
            }
        }
        return config

    def generate_env_variation(self, config, num_variations=5, save_to_file=False, **kwargs):
        """
        TODO: add more randomly generated configs instead of using manually specified configs. 
        """
        dim_xs = [5, 6, 7, 8]
        dim_zs = [5, 6, 7, 8]

        self.cached_configs = []
        self.cached_init_states = []

        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]
        for idx in range(num_variations):
            dim_x = random.choice(dim_xs)
            dim_z = random.choice(dim_zs)
            m = min(dim_x, dim_z)
            p = np.random.rand()
            water_radius = config['fluid']['radius'] * config['fluid']['rest_dis_coef']
            if p < 1. / 3.:  # small water volume
                print("small volume water")
                dim_y = 2 * m + np.random.randint(0, 2)
                v = dim_x * dim_y * dim_z
                h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius
                print("h {}".format(h))
                glass_height = h + (np.random.rand() - 0.5) * 0.05
            elif 1. / 3 < p and p < 2. / 3:  # midium water volumes
                print("medium volume water")
                dim_y = int(2.5 * m) + np.random.randint(0, 2)
                v = dim_x * dim_y * dim_z
                h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 1.5
                print("h {}".format(h))
                glass_height = h + (np.random.rand() - 0.5) * 0.05
            else:
                print("large volume water")
                dim_y = 4 * m
                v = dim_x * dim_y * dim_z
                h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 2.2
                print("h {}".format(h))
                glass_height = h + (m + np.random.rand()) * 0.007

            print("dim_x {} dim_y {} dim_z {} glass_height {}".format(dim_x, dim_y, dim_z, glass_height))
            config_variations[idx]['fluid']['dim_x'] = dim_x
            config_variations[idx]['fluid']['dim_y'] = dim_y
            config_variations[idx]['fluid']['dim_z'] = dim_z

            config_variations[idx]['glass']['height'] = glass_height

            self.set_scene(config_variations[idx])
            init_state = copy.deepcopy(self.get_state())

            self.cached_configs.append(config_variations[idx])
            self.cached_init_states.append(init_state)

        combined = [self.cached_configs, self.cached_init_states]

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        if self.observation_mode in ['point_cloud', 'key_point']:
            if self.observation_mode == 'key_point':
                obs_dim = 0
            else:
                obs_dim = pyflex.get_n_particles() * 3
                self.particle_obs_dim = obs_dim
            obs_dim += 1  # self.glass_x
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif self.observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        self.inner_step = 0
        return self._get_obs()

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'glass_x': self.glass_x, 'glass_states': self.glass_states, 'glass_params': self.glass_params}

    def set_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        self.glass_params = state_dic['glass_params']
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])
        self.glass_x = state_dic['glass_x']
        self.glass_states = state_dic['glass_states']
        for _ in range(5):
            pyflex.step()

    def initialize_camera(self):
        '''
        set the camera width, height, position and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        self.camera_params = {
            'default_camera': {'pos': np.array([2.2, 2.7, 0.3]),
                               'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
                               'width': self.camera_width,
                               'height': self.camera_height},
            'cam_2d': {'pos': np.array([0.5, .7, 4.]),
                       'angle': np.array([0, 0, 0.]),
                       'width': self.camera_width,
                       'height': self.camera_height}
        }

    def set_glass_params(self, config=None):
        params = config

        self.border = params['border']
        self.height = params['height']

        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.1  # glass floor length
        self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.1  # glass width

        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['glass_x_center'] = self.x_center

        self.glass_params = params

    def set_scene(self, config, states=None):
        '''
        Construct the pouring water scence.
        '''
        # create fluid
        super().set_scene(config["fluid"])  # do not sample fluid parameters, as it's very likely to generate very strange fluid
        # print("fluid particle num: ", pyflex.get_n_particles())

        # compute glass params
        self.set_glass_params(config["glass"])

        # create pouring glass & poured glass
        self.create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)

        # move pouring glass to be at ground
        self.glass_floor_centerx = self.x_center
        self.glass_states = self.init_glass_state(self.x_center, 0, self.glass_dis_x, self.glass_dis_z, self.height, self.border)

        pyflex.set_shape_states(self.glass_states)

        # record glass floor center x
        self.glass_x = self.x_center

        # no cached init states passed in 
        if states is None:
            fluid_pos = np.ones((self.particle_num, self.dim_position))
            # print(self.particle_num)

            # move water all inside pouring cup
            fluid_radius = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            fluid_dis = np.array([1.2 * fluid_radius, fluid_radius * 0.45, 1.2 * fluid_radius])
            lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 2.
            lower_z = -self.glass_params['glass_dis_z'] / 2 + 0.05
            lower_y = self.glass_params['border']
            lower = np.array([lower_x, lower_y, lower_z])
            cnt = 0
            for x in range(self.fluid_params['dim_x']):
                for y in range(self.fluid_params['dim_y']):
                    for z in range(self.fluid_params['dim_z']):
                        fluid_pos[cnt][:3] = lower + np.array([x, y, z]) * fluid_dis  # + np.random.rand() * 0.01
                        cnt += 1

            pyflex.set_positions(fluid_pos)
            print("stablize water!")
            for _ in range(300):
                pyflex.step()
        else:  # set to passed-in cached init states
            self.set_state(states)

        # print("pour water inital scene constructed over...")

    def _get_obs(self):
        '''
        return the observation based on the current flex state.
        '''
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_width, self.camera_height)
        elif self.observation_mode in ['point_cloud', 'key_point']:
            if self.observation_mode == 'point_cloud':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
                pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
                pos[:len(particle_pos)] = particle_pos
            else:
                pos = np.empty(0, dtype=np.float)

            cup_state = np.array([self.glass_x])
            return np.hstack([pos, cup_state]).flatten()
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """
        The reward is computed as the fraction of water in the poured glass and the distance to the target
        NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
        """
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        out_glass = water_num - in_glass

        # print("out glass water num: ", out_glass)
        reward = -self.water_penalty_coef * (out_glass / water_num)
        reward += -self.distance_coef * np.abs((self.terminal_x - self.glass_x))

        if set_prev_reward:
            delta_reward = reward - self.prev_reward
            self.prev_reward = reward

        return delta_reward if self.delta_reward else reward

    def compute_in_pouring_glass_water(self):
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape(-1, self.dim_position)
        in_pouring_glass = 0
        for water in water_state:
            res = self.in_glass(water, self.glass_states, self.border, self.height)
            in_pouring_glass += res
        return in_pouring_glass

    def _step(self, action):
        '''
        action: np.ndarray of dim 1x1, dx, which specifies how much to move on the x-axis.
        '''
        # make action as increasement, clip its range
        dx = action[0]
        dx = np.clip(dx, a_min=-self.border, a_max=self.border)
        x = self.glass_x + dx

        # move the glass
        new_states = self.move_glass(self.glass_states, x)
        self.glass_states = new_states
        self.glass_x = x
        self.glass_x = np.clip(self.glass_x, a_min=self.min_x, a_max=self.max_x)

        # pyflex takes a step to update the glass and the water fluid
        pyflex.set_shape_states(self.glass_states)
        pyflex.step()

        self.inner_step += 1
        return

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

    def move_glass(self, prev_states, x):
        '''
        given the previous states of the glass, move it in 1D along x-axis.
        update the states of the 5 boxes that form the box: floor, left/right wall, back/front wall. 
        
        state:
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat 
        10-14: previous quat 
        '''
        dis_x, dis_z = self.glass_dis_x, self.glass_dis_z
        quat_curr = quatFromAxisAngle([0, 0, -1.], 0.)

        border = self.border

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        for i in range(5):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]

        x_center = x
        y = 0

        # floor: center position does not change
        states[0, :3] = np.array([x_center, y, 0.])

        # left wall: center must move right and move down. 
        relative_coord = np.array([-(dis_x + border) / 2., (self.height + border) / 2., 0.])
        states[1, :3] = states[0, :3] + relative_coord

        # right wall
        relative_coord = np.array([(dis_x + border) / 2., (self.height + border) / 2., 0.])
        states[2, :3] = states[0, :3] + relative_coord

        # back wall
        relative_coord = np.array([0, (self.height + border) / 2., -(dis_z + border) / 2.])
        states[3, :3] = states[0, :3] + relative_coord

        # front wall
        relative_coord = np.array([0, (self.height + border) / 2., (dis_z + border) / 2.])
        states[4, :3] = states[0, :3] + relative_coord

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
        states[1, :3] = np.array([x_center - (dis_x + border) / 2., (height + border) / 2. + y_curr, 0.])
        states[1, 3:6] = np.array([x_center - (dis_x + border) / 2., (height + border) / 2. + y_last, 0.])

        # right wall
        states[2, :3] = np.array([x_center + (dis_x + border) / 2., (height + border) / 2. + y_curr, 0.])
        states[2, 3:6] = np.array([x_center + (dis_x + border) / 2., (height + border) / 2. + y_last, 0.])

        # back wall
        states[3, :3] = np.array([x_center, (height + border) / 2. + y_curr, -(dis_z + border) / 2.])
        states[3, 3:6] = np.array([x_center, (height + border) / 2. + y_last, -(dis_z + border) / 2.])

        # front wall
        states[4, :3] = np.array([x_center, (height + border) / 2. + y_curr, (dis_z + border) / 2.])
        states[4, 3:6] = np.array([x_center, (height + border) / 2. + y_last, (dis_z + border) / 2.])

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states

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
        res = np.sum(res)
        return res

    def _get_info(self):
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        out_glass = water_num - in_glass
        reward = -self.water_penalty_coef * (out_glass / water_num)
        reward += -self.distance_coef * np.abs((self.terminal_x - self.glass_x))
        return {'performance': reward,
                'distance_to_target': np.abs((self.terminal_x - self.glass_x)),
                'out_water': (out_glass / water_num)}


if __name__ == '__main__':
    env = PassWater1DEnv(observation_mode='cam_rgb',
                         action_mode='direct',
                         render=True,
                         headless=False,
                         horizon=75,
                         action_repeat=8,
                         render_mode='fluid',
                         deterministic=True)
    env.reset()
    for i in range(500):
        pyflex.step()
