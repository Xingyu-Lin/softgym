import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv


class ClothFlattenEnv(ClothEnv):
    def __init__(self, cached_init_state_path='cloth_flatten_init_states.pkl', **kwargs):
        """
        :param cached_init_state_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(config_file="ClothFlattenConfig.yaml", **kwargs)
        self.prev_covered_area = None  # Should not be used until initialized
        self.cached_init_state = []

        if cached_init_state_path.startswith('/'):
            self.cached_init_state_path = cached_init_state_path
        else:
            cur_dir = osp.dirname(osp.abspath(__file__))
            self.cached_init_state_path = osp.join(cur_dir, cached_init_state_path)
        if osp.exists(self.cached_init_state_path):
            self._load_init_state()
            print('ClothFlattenEnv: {} cached initial states loaded'.format(len(self.cached_init_state)))

    def initialize_camera(self):
        '''
        set the camera width, height, ition and angle.
        **Note: width and height is actually the screen width and screen height of FLex.
        I suggest to keep them the same as the ones used in pyflex.cpp.
        '''
        self.camera_params = {
            'pos': np.array([0.5, 3, 2.5]),
            'angle': np.array([0, -50 / 180. * np.pi, 0.]),
            'width': self.camera_width,
            'height': self.camera_height
        }

    def generate_init_state(self, num_init_state=1, save_to_file=True):
        """ Generate initial states. Note: This will also change the current states! """
        # TODO Xingyu: Add options for generating initial states with different parameters. Currently only the pickpoint varies.
        # TODO additionally, can vary the height / number of pick point
        original_state = self.get_state()
        num_particle = original_state['particle_pos'].reshape((-1, 4)).shape[0]
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        init_states = []

        for i in range(num_init_state):
            pickpoint = random.randint(0, num_particle)
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
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

            if self.action_mode == 'sphere' or self.action_mode == 'picker':
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])

            init_states.append(self.get_state())
            self.set_state(original_state)

        if save_to_file:
            with open(self.cached_init_state_path, 'wb') as handle:
                pickle.dump(init_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return init_states

    def _load_init_state(self):
        with open(self.cached_init_state_path, "rb") as handle:
            self.cached_init_state = pickle.load(handle)

    def _reset(self):
        """ Right now only use one initial state"""
        if len(self.cached_init_state) == 0:
            state_dicts = self.generate_init_state(1)
            self.cached_init_state.extend(state_dicts)
        cached_id = np.random.randint(len(self.cached_init_state))
        self.set_state(self.cached_init_state[cached_id])
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())

        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0, 1, 0])
        pyflex.step()
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
            for _ in range(self.action_repeat):
                pyflex.step()
                self.action_tool.step(action)
        pos = pyflex.get_positions()
        reward = self.compute_reward(pos)
        return self._get_obs(), reward, False, {}

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

    def compute_reward(self, particle_pos):
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area - self.prev_covered_area
        self.prev_covered_area = curr_covered_area
        return r
