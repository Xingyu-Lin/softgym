import numpy as np
from gym.spaces import Box

import pyflex
from softgym.envs.fluid_rigid_env import FluidTorusEnv
import copy
from softgym.utils.misc import quatFromAxisAngle
from softgym.action_space.robot_env import RobotBase
import pickle
import os.path as osp


class TransportTorus1D(FluidTorusEnv):
    def __init__(self, observation_mode, action_mode, config=None, cached_states_path='transport_torus_init_states.pkl', **kwargs):
        '''
        This class implements a transport torus task.
        The torus is put on a box. You need to move the box to a target location.
        This is a 1D task.

        observation_mode: "cam_rgb" or "full_state"
        action_mode: "direct"
        
        TODO: add more description of the task.
        '''
        assert observation_mode in ['cam_rgb', 'point_cloud', 'key_point']

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.wall_num = 5  # number of box walls. floor/left/right/front/back
        self.distance_coef = 1.
        self.torus_penalty_coef = 10.
        self.terminal_x = 1.2
        self.min_x = -0.25
        self.max_x = 1.4
        self.prev_reward = 0

        self.reward_min = self.distance_coef * min(self.min_x - self.terminal_x, self.terminal_x - self.max_x) - \
            self.torus_penalty_coef * 1
        self.reward_max = 0 # reach target position, with no water spilled
        self.reward_range = self.reward_max - self.reward_min

        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        if observation_mode in ['point_cloud', 'key_point']:
            if observation_mode == 'key_point':
                obs_dim = 0
            else:
                max_particle_num = 13 * 13 * 13 * 4
                obs_dim = max_particle_num * 3
                self.particle_obs_dim = obs_dim
            # z and theta of the second cup (poured_box) does not change and thus are omitted.
            obs_dim += 8      # Pos (x) and shape (w, h, l) reset of the cup, torus x, torus y, if on box.
            self.observation_space = Box(low=np.array([-np.inf] * obs_dim), high=np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)

        default_config = self.get_default_config()
        if action_mode == 'direct':
            self.action_direct_dim = 1
            action_low = np.array([-0.011])
            action_high = np.array([0.011])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)
        else:
            raise NotImplementedError

    def get_default_config(self):
        config = {
            'torus': {
                'radius': 0.03, # if self.action_mode in ['sawyer', 'franka'] else 0.1,
                'rest_dis_coef': 0.5,
                'num': 5,
                'size': 0.2,
            },
            'box': { # all these will be overwritten by generate_env_variations
                'box_dis_x': 0.6,
                'box_dis_z': 0.6,
                'height': 0.6, 
            },
            'static_friction': 0.5,
            'dynamic_friction': 1.0,
            'camera_name': 'default_camera',
        }
        return config

    def generate_env_variation(self, num_variations=5,  **kwargs):
        """
        TODO: add more randomly generated configs instead of using manually specified configs. 
        """
        num_low = 1
        num_high = 2
        size_low = 0.12
        size_high = 0.28
        box_height_low = 0.1
        box_height_high = 0.3
        generated_configs = []
        generated_init_states = []
        config = self.get_default_config()
        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]
        
        idx = 0
        while idx < num_variations:
            num = np.random.randint(num_low, num_high)
            size = np.random.uniform(size_low, size_high)
            
            particle_radius = config['torus']['radius'] * config['torus']['rest_dis_coef']
            estimated_particle_num_height = int(size / 0.05 -1)
            print("estimated_particle_num_height: ", estimated_particle_num_height)
            height = num * estimated_particle_num_height * particle_radius + num * size * 4 * particle_radius
            
            estimated_particle_num_width = int(size / 0.05 * 3 + 2)
            box_dis_x = estimated_particle_num_width * particle_radius + particle_radius  # box floor length
            box_dis_z = estimated_particle_num_width * particle_radius + particle_radius  # box width

            config_variations[idx]['torus']['lower_x'] = - box_dis_x / 2.
            config_variations[idx]['torus']['lower_z'] = - box_dis_z / 2.

            print("num {} size {}".format(num, config['torus']['size']))
            config_variations[idx]['torus']['num'] = num
            config_variations[idx]['torus']['size'] = size

            config_variations[idx]['box']['height'] = np.random.uniform(box_height_low, box_height_high)
            config_variations[idx]['torus']['height'] = estimated_particle_num_height * particle_radius + \
                config_variations[idx]['box']['height']
            config_variations[idx]['box']['box_dis_x'] = box_dis_x
            config_variations[idx]['box']['box_dis_z'] = box_dis_z
            print("box x, y, z: ", box_dis_x, height, box_dis_z)

            if self.set_scene(config_variations[idx]):
                init_state = copy.deepcopy(self.get_state())

                generated_configs.append(config_variations[idx])
                generated_init_states.append(init_state)
                idx += 1

        combined = [generated_configs, generated_init_states]

        return generated_configs, generated_init_states

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
        return self._get_obs()

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'box_x': self.box_x, 'box_states': self.box_states, 'box_params': self.box_params, 'config_id': self.current_config_id}

    def set_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        self.box_params = state_dic['box_params']
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])
        self.box_x = state_dic['box_x']
        self.box_states = state_dic['box_states']
        for _ in range(5):
            pyflex.step()

    def initialize_camera(self):
        '''
        set the camera width, height, position and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        # if self.action_mode in ['sawyer', 'franka']:
        self.camera_params = {
            'default_camera': {'pos': np.array([0.64, 1.8, 0.85]),
                            'angle': np.array([0 * np.pi, -65 / 180. * np.pi, 0]),
                            'width': self.camera_width,
                            'height': self.camera_height},
            'cam_2d': {'pos': np.array([0.5, .7, 4.]),
                    'angle': np.array([0, 0, 0.]),
                    'width': self.camera_width,
                    'height': self.camera_height}
        }
        # else:
        #     self.camera_params = {
        #     'default': {'pos': np.array([2.2, 2.7, 0.3]),
        #                        'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
        #                        'width': self.camera_width,
        #                        'height': self.camera_height},
        #     'cam_2d': {'pos': np.array([0.5, .7, 4.]),
        #                'angle': np.array([0, 0, 0.]),
        #                'width': self.camera_width,
        #                'height': self.camera_height}
        # }

    def set_box_params(self, config=None):
        params = config['box']

        self.height = params['height']

        # TODO: correctly determine the box size
        particle_radius = config['torus']['radius'] * config['torus']['rest_dis_coef']
        self.box_dis_x = params['box_dis_x']
        self.box_dis_z = params['box_dis_z']
        
        self.x_center = 0
        params['box_x_center'] = 0

        self.box_params = params

    def set_scene(self, config, states=None):
        '''
        Construct the passing water scence.
        '''
        # create fluid
        super().set_scene(config)  # do not sample fluid parameters, as it's very likely to generate very strange fluid

        # compute box params
        if states is None:
            self.set_box_params(config)
        else:
            box_params = states['box_params']
            self.height = box_params['height']
            self.box_dis_x = box_params['box_dis_x']
            self.box_dis_z = box_params['box_dis_z']
            self.box_params = box_params
            self.x_center = 0

        # create box
        self.create_box(self.box_dis_x, self.box_dis_z, self.height)

        # move box to be at ground or on the table
        self.box_states = self.init_box_state(self.x_center, 0, self.box_dis_x, self.box_dis_z, self.height)

        pyflex.set_shape_states(self.box_states)

        # record box floor center x
        self.box_x = self.x_center

        # no cached init states passed in 
        if states is None:
            for _ in range(50):
                pyflex.step()
                pyflex.render()

            return True

        else:  # set to passed-in cached init states
            self.set_state(states)


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

            particle_state = pyflex.get_positions().reshape((-1, self.dim_position))
            torus_center_y = np.mean(particle_state[:, 1])
            torus_center_x = np.mean(particle_state[:, 0])

            on_box = float(torus_center_y >= self.height)
            within_box_bound = float(torus_center_x > self.box_x - 0.5 * self.box_dis_x and 
                torus_center_x < self.box_x + self.box_dis_x * 0.5)
            cup_state = np.array([self.box_x, self.box_dis_x, self.box_dis_z, self.height, 
                torus_center_x, torus_center_y, on_box, within_box_bound])
            return np.hstack([pos, cup_state]).flatten()
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """
        The reward is the negative distance to the target.
        If the torus falls off the box, give a large negative penalty.
        """

        state_dic = self.get_state()
        particle_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        torus_center_y = np.mean(particle_state[:, 1])
        torus_center_x = np.mean(particle_state[:, 0])

        reward = -self.distance_coef * np.abs((self.terminal_x - self.box_x))
        box_x_low = self.box_x - self.box_dis_x / 2.
        box_x_high = self.box_x + self.box_dis_x / 2.
        if torus_center_y < self.height or torus_center_x < box_x_low or torus_center_x > box_x_high:
            reward -= self.torus_penalty_coef
        return reward

    def _get_info(self):
        state_dic = self.get_state()
        particle_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        torus_center_y = np.mean(particle_state[:, 1])
        torus_center_x = np.mean(particle_state[:, 0])

        reward = -self.distance_coef * np.abs((self.terminal_x - self.box_x))
        box_x_low = self.box_x - self.box_dis_x / 2.
        box_x_high = self.box_x + self.box_dis_x / 2.
        if torus_center_y < self.height or torus_center_x < box_x_low or torus_center_x > box_x_high:
            reward -= self.torus_penalty_coef

        performance = reward
        performance_init =  performance if self.performance_init is None else self.performance_init  # Use the original performance
        normalized_performance = (performance - performance_init) / (self.reward_max - performance_init)

        return {'performance': performance,
                'normalized_performance': normalized_performance,
                'distance_to_target': np.abs((self.terminal_x - self.box_x)),
                'torus_on': float(torus_center_y >= self.height)}

    def _step(self, action):
        '''
        action: np.ndarray of dim 1x1, dx, which specifies how much to move on the x-axis.
        '''
        # make action as increasement, clip its range
        dx = action[0]
        dx = np.clip(dx, a_min=self.action_space.low[0], a_max=self.action_space.high[0])
        x = self.box_x + dx

        # move the box
        new_states = self.move_box(self.box_states, x)
        self.box_states = new_states
        self.box_x = x
        self.box_x = np.clip(self.box_x, a_min=self.min_x, a_max=self.max_x)

        # pyflex takes a step to update the box and the water fluid
        pyflex.set_shape_states(self.box_states)
        pyflex.step()

        self.inner_step += 1

    def create_box(self, box_dis_x, box_dis_z, height):
        """
        create a box.
        dis_x: the length of the box
        dis_z: the width of the box
        height: the height of the box.

        the halfEdge determines the center point of each wall.
        """
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        boxes = []

        # a single box
        halfEdge = np.array([box_dis_x / 2., height / 2., box_dis_z / 2.])
        boxes.append([halfEdge, center, quat])

        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)

        return boxes

    def move_box(self, prev_states, x):
        '''
        given the previous states of the box, move it in 1D along x-axis.
        
        state:
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat 
        10-14: previous quat 
        '''
        quat_curr = quatFromAxisAngle([0, 0, -1.], 0.)
        states = np.zeros((1, self.dim_shape_state))

        for i in range(1):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]

        x_center = x
        y = self.height / 2.

        # floor: center position does not change
        states[0, :3] = np.array([x_center, y, 0.])
        states[:, 6:10] = quat_curr

        return states

    def init_box_state(self, x, y, box_dis_x, box_dis_z, height):
        '''
        set the initial state of the box.
        '''
        dis_x, dis_z = box_dis_x, box_dis_z
        x_center, y_curr, y_last = x, y, 0.
        if self.action_mode in ['sawyer', 'franka']:
            y_curr = y_last = 0.56 # NOTE: robotics table
        quat = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of a single box
        states = np.zeros((1, self.dim_shape_state))

        # a single box
        states[0, :3] = np.array([x_center, height / 2., 0.])
        states[0, 3:6] = np.array([x_center, height / 2., 0.])

        states[:, 6:10] = quat
        states[:, 10:] = quat

        return states

    def in_box(self, water, box_states, border, height, return_sum=True):
        '''
        judge whether a water particle is in the poured box
        water: [x, y, z, 1/m] water particle state.
        '''
        # floor, left, right, back, front
        # state:
        # 0-3: current (x, y, z) coordinate of the center point
        # 3-6: previous (x, y, z) coordinate of the center point
        # 6-10: current quat 
        # 10-14: previous quat 
        x_lower = box_states[1][0] - border / 2.
        x_upper = box_states[2][0] + border / 2.
        z_lower = box_states[3][2] - border / 2.
        z_upper = box_states[4][2] + border / 2
        # y_lower = box_states[0][1] - border / 2.
        # y_upper = box_states[0][1] + height + border / 2.
        # x, y, z = water[:, 0], water[:, 1], water[:, 2]
        x, z = water[:, 0], water[:, 2]

        # * (y >= y_lower) * (y <= y_upper)
        res = (x >= x_lower) * (x <= x_upper) * (z >= z_lower) * (z <= z_upper)
        if return_sum:
            res = np.sum(res)
            return res
        return res

    def _get_current_water_height(self):
        pos = pyflex.get_positions().reshape(-1, 4)
        return np.max(pos[:, 1])