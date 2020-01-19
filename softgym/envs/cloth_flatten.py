import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv
import copy
from copy import deepcopy


class ClothFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_flatten_init_states.pkl', num_variations = 2, **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.prev_covered_area = None  # Should not be used until initialized
        self.num_variations = num_variations
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_states_path = osp.join(cur_dir, cached_states_path)
        else:
            self.cached_states_path = cached_states_path
        success = self.get_cached_configs_and_states(cached_states_path)
        if not success or not self.use_cached_states:
            self.generate_env_variation(num_variations, save_to_file=True)
            success = self.get_cached_configs_and_states(cached_states_path)
            assert success

    def initialize_camera(self, make_multitask_happy=None):
        """
        set the camera width, height, ition and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        """
        self.camera_name = 'default_camera'
        self.camera_params['default_camera'] = {
            'pos': np.array([0., 4., 0.]),
            'angle': np.array([0, -70 / 180. * np.pi, 0.]),
            'width': self.camera_width,
            'height': self.camera_height
        }

    def _sample_cloth_size(self):
        return np.random.randint(32), np.random.randint(64)

    def generate_env_variation(self, num_variations=1, save_to_file=False, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        # TODO Xingyu: Add options for generating initial states with different parameters. Currently only the pickpoint varies.
        # TODO additionally, can vary the height / number of pick point
        # original_state = self.get_state()
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        for i in range(num_variations):
            config = deepcopy(default_config)
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
                self.set_scene(config)
                self.action_tool.reset([0., -1., 0.])
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            num_particle = cloth_dimx * cloth_dimy
            pickpoint = random.randint(0, num_particle-1)
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
            pickpoint_pos[1] += np.random.random(1) * 0.5
            pyflex.set_positions(curr_pos)

            # Pick up the cloth and wait to stablize
            for _ in range(0, max_wait_step):
                pyflex.step()
                curr_pos = pyflex.get_positions()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)

            # Drop the cloth and wait to stablize
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = 1
            pyflex.set_positions(curr_pos)
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break
            curr_pos = pyflex.get_positions()
            camera_param = copy.copy(self.camera_params[self.camera_name])
            cx, cy = self._get_center_point(curr_pos)
            camera_param['pos'][0] = float(cx)
            camera_param['pos'][2] = float(cy) + 1.5
            self.update_camera(self.camera_name, camera_param)
            config['camera_params'] = deepcopy(self.camera_params)

            if self.action_mode == 'sphere' or self.action_mode == 'picker':
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))
            # self.set_state(original_state)

        if save_to_file:
            with open(self.cached_states_path, 'wb') as handle:
                pickle.dump((generated_configs, generated_states), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return generated_configs, generated_states

    def _reset(self):
        """ Right now only use one initial state"""
        # if len(self.cached_init_state) == 0:
        #     state_dicts = self.generate_init_state(1)
        #     self.cached_init_state.extend(state_dicts)
        # cached_id = np.random.randint(len(self.cached_init_state))
        # self.set_state(self.cached_init_state[cached_id])
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0, 1, 0])
        return self._get_obs()

    def _step(self, action):
        if self.action_mode.startswith('key_point'):
            # TODO ad action_repeat
            print('Need to add action repeat')
            raise NotImplementedError
            valid_idxs = np.array([0, 63, 31 * 64, 32 * 64 - 1])
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            pyflex.step()

            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            action = action.reshape([-1, 4])
            idxs = np.hstack(action[:, 0])
            updates = action[:, 1:]
            action = np.hstack([action, np.zeros([action.shape[0], 1])])
            vels = pyflex.get_velocities()
            cur_pos[:, 3] = 1
            if self.action_mode == 'key_point_pos':
                cur_pos[valid_idxs[idxs.astype(int)], :3] = last_pos[valid_idxs[idxs.astype(int)]][:, :3] + updates
                cur_pos[valid_idxs[idxs.astype(int)], 3] = 0


            else:
                vels = np.array(vels).reshape([-1, 3])
                vels[idxs.astype(int), :] = updates
            pyflex.set_positions(cur_pos.flatten())
            pyflex.set_velocities(vels.flatten())
        else:
            pyflex.step()
            self.action_tool.step(action)
        return

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

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area - self.prev_covered_area
        self.prev_covered_area = curr_covered_area
        return r
