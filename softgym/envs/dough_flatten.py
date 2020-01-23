import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.dough_env import DoughEnv
import scipy
import copy
from copy import deepcopy

from softgym.envs.util import quatFromAxisAngle


class DoughFlattenEnv(DoughEnv):
    def __init__(self, cached_states_path='dough_flatten_init_states.pkl', num_variations=2, **kwargs):
        """
        :param cached_states_path: path to the cached init states
        """

        super().__init__(**kwargs)
        self.prev_covered_area = None  # Should not be used until initialized
        self.num_variations = num_variations

        # TODO: add task variations
        # if not cached_states_path.startswith('/'):
        #     cur_dir = osp.dirname(osp.abspath(__file__))
        #     self.cached_states_path = osp.join(cur_dir, cached_states_path)
        # else:
        #     self.cached_states_path = cached_states_path
        # success = self.get_cached_configs_and_states(cached_states_path)
        # if not success or not self.use_cached_states:
        #     self.generate_env_variation(num_variations, save_to_file=True)
        #     success = self.get_cached_configs_and_states(cached_states_path)
        #     assert success

        self.action_low = -0.05
        self.action_high = 0.05
        self.cached_configs = [self.get_default_config()]
        self.cached_init_states = [None]

    def get_default_config(self):
        config = DoughEnv.get_default_config(self)
        config['capsule'] = {
            'radius': 0.1,
            'halfheight': 0.6,
        }
        return config

    def set_scene(self, config, state=None):

        # first add a sphere dough
        DoughEnv.set_scene(self, config)

        # add a capsule
        self.capsule_radius = config['capsule']['radius']
        self.capsule_halfheight = config['capsule']['halfheight']
        self.capsule_params = deepcopy(config['capsule'])
        self.create_capsule(self.capsule_radius, self.capsule_halfheight)

        if state is not None:
            self.set_state(state)
    
    def get_state(self):
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'capsule_x': self.capsule_x, 'capsule_y': self.capsule_y, 'capsule_z': self.capsule_z,
                'capsule_rotation': self.capsule_rotation, 'capsule_params': self.capsule_params, 
                'capsule_states': self.capsule_states}

    def set_state(self, state_dic):
        # rebuild the capsule according to the capsule params
        self.capsule_radius = state_dic['capsule_params']['capsule_radius']
        self.capsule_halfheight = state_dic['capsule_params']['capsule_halfheight']

        pyflex.pop_box(self.wall_num) # TODO: change this to pop capsule
        self.create_capsule(self.capsule_radius, self.capsule_halfheight)

        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])
        self.capsule_x = state_dic['capsule_x']
        self.capsule_y = state_dic['capsule_y']
        self.capsule_z = state_dic['capsule_z']
        self.capsule_rotation = state_dic['capsule_rotation']
        self.capsule_states = state_dic['capsule_states']
        for _ in range(5):
            pyflex.step()

    def create_capsule(self, radius=0.2, halfheight=0.5):
        particle_pos = pyflex.get_positions().reshape((-1, self.dim_position))
        min_x, min_y, min_z = np.min(particle_pos[:, 0]), np.min(particle_pos[:, 1]), np.min(particle_pos[:, 2])
        max_x, max_y, max_z = np.max(particle_pos[:, 0]), np.max(particle_pos[:, 1]), np.max(particle_pos[:, 2])
        params = np.array([radius, halfheight])
        lower = np.array([min_x, max_y, min_z])
        quat = quatFromAxisAngle([0, -1, 0], 0.)
        pyflex.add_capsule(params, lower, quat)

        self.capsule_x = min_x + 0.1
        self.capsule_y = max_y + 0.3
        self.capsule_z = min_z + 0.1
        self.capsule_rotation = 0.
        self.capsule_pos = []
        self.capsule_states = pyflex.get_shape_states()
        self.capsule_boundary = np.array([[min_x - 0.5, max_x + 0.5], [0.0, max_y + 0.5], [min_z - 0.5, max_z + 0.5]])

    def generate_env_variation(self, num_variations=1, save_to_file=False, **kwargs):
        """ Generate initial states. Note: This will also change the current states! """
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        for i in range(num_variations):
            config = deepcopy(default_config)
            self.set_scene(config)

            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, 3] = config['ParticleInvMass']
            pyflex.set_positions(pos)

            self.update_camera('default_camera', default_config['camera_params']['default_camera'])
            config['camera_params'] = deepcopy(self.camera_params)
            self.action_tool.reset([0., -1., 0.])

            self._random_pick_and_place(pick_num=10)
            self._center_object()
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return generated_configs, generated_states

    def _reset(self):
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        return self._get_obs()

    def _step(self, action):
        move = action[:3]
        rotate = action[3]
        move = np.clip(move, a_min=self.action_low, a_max=self.action_high)
        rotate = np.clip(rotate, a_min=self.action_low, a_max=self.action_high)
        dx, dy, dz, dtheta = move[0], move[1], move[2], rotate
        print("currnet y {} dx {} dy {} dz {} dtheta {}".format(self.capsule_y, dx, dy, dz, dtheta))
        x, y, z, theta = self.capsule_x + dx, self.capsule_y + dy, self.capsule_z + dz, self.capsule_rotation + dtheta
        y = max(0, y) # make sure capsule is above ground

        # check if the movement of the pouring glass collide with the poured glass.
        # the action only take effects if there is no collision
        self.capsule_states = self.move_capsule(self.capsule_states, x, y, z, theta)
        self.capsule_x, self.capsule_y, self.capsule_z, self.capsule_rotation = x, y, z, theta
        self._apply_capsule_boundary()

        # pyflex takes a step to update the dough and
        pyflex.set_shape_states(self.capsule_states)
        pyflex.step()
        
        return

    def move_capsule(self, prev_states, x, y, z, theta):
        quat_curr = quatFromAxisAngle([0, -1, 0], theta)
        states = np.zeros(self.dim_shape_state)

        states[3:6] = prev_states[:3]
        states[10:] = prev_states[6:10]
        states[:3] = np.array([x, y, z])
        states[6:10] = quat_curr

        return states

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area - self.prev_covered_area
        if set_prev_reward:
            self.prev_covered_area = curr_covered_area
        return r

    @staticmethod
    def _get_current_covered_area(pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        grid = np.zeros([101, 101])  # Discretization
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]
        offset = pos2d - init
        slotted_x = (offset[:, 0] // span[0])
        slotted_y = (offset[:, 1] // span[1])
        grid[slotted_y.astype(int), slotted_x.astype(int)] = 1
        return np.sum(grid) * span[0] * span[1]

    def _apply_capsule_boundary(self):
        self.capsule_x = np.clip(self.capsule_x, self.capsule_boundary[0][0], self.capsule_boundary[0][1])
        self.capsule_y = np.clip(self.capsule_y, self.capsule_boundary[1][0], self.capsule_boundary[1][1])
        self.capsule_z = np.clip(self.capsule_z, self.capsule_boundary[2][0], self.capsule_boundary[2][1])


if __name__ == '__main__':
    env = DoughFlattenEnv(observation_mode='full_state',
                  action_mode='direct',
                  render=True,
                  headless=False,
                  horizon=75,
                  action_repeat=8,
                  num_variations=200,
                  use_cached_states=True,
                  deterministic=False)

    env.reset()
    for i in range(500):
        pyflex.step()
