import numpy as np
from gym.spaces import Box
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv
from softgym.envs.action_space import ParallelGripper, Picker


class ClothFlattenPointControlEnv(ClothEnv):
    def __init__(self, observation_mode, action_mode, horizon=250, cached_init_state_path='cloth_flatten_init_states.pkl',
                 num_picker=2, **kwargs):
        """

        :param observation_mode:
        :param action_mode:
        :param horizon:
        :param cached_init_state_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(config_file="ClothFlattenConfig.yaml", **kwargs)
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point_pos', 'key_point_vel', 'sphere', 'picker']

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.video_path = "data/videos"
        self.prev_reward = 0

        self.horizon = horizon

        if action_mode.startswith('key_point'):
            space_low = np.array([0, -0.1, -0.1, -0.1] * 2)
            space_high = np.array([3.9, 0.1, 0.1, 0.1] * 2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode.startswith('sphere'):
            self.action_tool = ParallelGripper(gripper_type='sphere')
            self.action_space = self.action_tool.action_space
        elif action_mode == 'picker':
            self.action_tool = Picker(num_picker)
            self.action_space = self.action_tool.action_space

        if observation_mode == 'key_point':  # TODO: add sphere position
            if action_mode == 'key_point_pos':
                self.observation_space = Box(np.array([-np.inf] * pyflex.get_n_particles() * 3),
                                             np.array([np.inf] * pyflex.get_n_particles() * 3), dtype=np.float32)
            elif action_mode == 'sphere':
                # TODO observation space should depend on the action_tool
                self.observation_space = Box(np.array([-np.inf] * (pyflex.get_n_particles() * 3 + 4 * 3)),
                                             np.array([np.inf] * (pyflex.get_n_particles() * 3 + 4 * 3)), dtype=np.float32)
            elif action_mode == 'picker':
                self.observation_space = Box(np.array([-np.inf] * (pyflex.get_n_particles() * 3 + num_picker * 3)),
                                             np.array([np.inf] * (pyflex.get_n_particles() * 3 + num_picker * 3)), dtype=np.float32)
        else:
            raise NotImplementedError

        self.storage_name = "test_flatten"
        self.video_height = 240
        self.video_width = 320

        self.prev_covered_area = None  # Should not be used until initialized
        self.cached_init_state = []
        if cached_init_state_path is not None:
            self._load_init_state(cached_init_state_path)

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

    def generate_init_state(self, num_init_state=1, save_to_file=False):
        """ Generate initial states. Note: This will also change the current states! """
        # TODO Xingyu: Add options for generating initial states with different parameters. Currently only the pickpoint varies.
        # TODO additionally, can vary the height / number of pick point
        original_state = self.get_state()
        num_particle = original_state['particle_pos'].reshape((-1, 4)).shape[0]
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.03  # Cloth stable when all particles' vel are smaller than this
        init_states = []

        for _ in range(num_init_state):
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

            if self.action_mode.startswith('sphere'):  # Add gripper
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])

            init_states.append(self.get_state())
            self.set_state(original_state)

        if save_to_file:
            cur_dir = osp.dirname(osp.abspath(__file__))
            with open(osp.join(cur_dir, 'cloth_flatten_init_states.pkl'), 'wb') as handle:
                pickle.dump(init_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return init_states

    def _load_init_state(self, init_state_path):
        cur_dir = osp.dirname(osp.abspath(__file__))
        with open(osp.join(cur_dir, init_state_path), "rb") as handle:
            self.cached_init_state = pickle.load(handle)

    def reset(self):
        """ Right now only use one initial state"""
        if len(self.cached_init_state) == 0:
            state_dicts = self.generate_init_state(1)
            self.cached_init_state.extend(state_dicts)
        cached_id = np.random.randint(len(self.cached_init_state))
        self.set_state(self.cached_init_state[cached_id])
        self.prev_covered_area = self._get_current_covered_area()

        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0, 1, 0])
        pyflex.step()
        return self._get_obs()

    def _get_obs(self):  # NOTE: just rename to _get_obs
        pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        if self.action_mode == 'sphere':
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            return np.concatenate([pos[:, :3].flatten(), shapes[:, 0:3].flatten()])
        else:
            return pos[:, :3].flatten()

    def set_video_recording_params(self):
        """
        Set the following parameters if video recording is needed:
            video_idx_st, video_idx_en, video_height, video_width
        """
        self.video_height = 240
        self.video_width = 320

    def _step(self, action):
        if self.action_mode.startswith('key_point'):
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
        return self._get_obs(), self.compute_reward(), False, {}

    @staticmethod
    def _get_current_covered_area():
        """ Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points"""
        pos = pyflex.get_positions()
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

    def compute_reward(self):
        curr_covered_area = self._get_current_covered_area()
        r = curr_covered_area - self.prev_covered_area
        self.prev_covered_area = curr_covered_area
        return r


"""
class ClothFlattenSphereControlEnv(ClothEnv):
    def __init__(self, observation_mode, action_mode):
        super().__init__()
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point_pos', 'key_point_vel']

        self.observation_mode = observation_mode
        self.action_mode = action_mode

        if observation_mode == 'key_point':
            self.observation_space = Box(np.array([-np.inf] * pyflex.get_n_particles()), np.array([np.inf] * pyflex.get_n_particles()), dtype=np.float32)
        else:
            raise NotImplementedError

        if action_mode.startswith('key_point'):

            self.action_space = Box(np.array([-0.1]*6), np.array([0.1]*6), dtype=np.float32)
        else:
            raise NotImplementedError
        self.storage_name = "test_flatten_spheres"
        self.reset_time_step = 500
        self.init_state = self.get_state()
        self.initPos = None
        self.initVel = None

    def reset(self, dropPoint=100, xdmin = 64, ydim = 32):
        self.i = 0
        if self.initPos is not None:
            print("resetting")
            pyflex.set_positions(self.initPos)
            pyflex.set_positions(self.initVel)
            pyflex.set_shape_states(self.initState)
            return
        pickpoint = random.randint(0, xdmin * ydim)
        if dropPoint is not None:
            pickpoint = dropPoint
        pyflex.set_scene(10, np.array([pickpoint, xdmin, ydim]), 0)

        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)
        print("pick point: {}".format(pickpoint))
        particle_pos = pyflex.get_positions()[pickpoint * 4: pickpoint * 4 + 3]
        stopParticle = True
        for i in range(0, self.reset_time_step):
            pyflex.step()
            newPos = pyflex.get_positions()
            vels = pyflex.get_velocities()
            if stopParticle:
                newPos[pickpoint * 4: pickpoint * 4 + 3] = particle_pos
                #newPos[pickpoint * 4 + 3] = 0.0
                vels[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            stopped = True
            for j in range(pyflex.get_n_particles()):
                if vels[j] > 0.2:
                    print("stopped check at {} with vel {}".format(j, vels[j]))
                    stopped = False
                    break
            if stopped:
                newPos[pickpoint * 4 + 3] = 1
                stopParticle = False

            pyflex.set_velocities(vels)
            pyflex.set_positions(newPos)
        print("dropping")
        lastPos = pyflex.get_positions()
        lastPos[pickpoint*4+3] = 1
        pyflex.set_positions(lastPos)
        for i in range (0, 100):
            pyflex.step()
        self.initPos = pyflex.get_positions()
        self.initVel = pyflex.get_velocities()

        pyflex.add_sphere(0.1, [1.5, 0.25, 2.5], [1, 0, 0, 0])
        pyflex.add_sphere(0.1, [0.0, 0.25, 2], [1, 0, 0, 0])
        self.initPos = pyflex.get_positions()
        self.initVel = pyflex.get_velocities()
        self.initState = pyflex.get_shape_states()
        #pyflex.set_scene(9, np.array([]), 0)
        #pyflex.set_positions(self.initPos)

    def _get_obs(self):
        pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        return pos[:, :3].flatten()

    def step(self, action):

        last_pos = np.array(pyflex.get_shape_states())
        print("shape: {}".format(last_pos.shape))
        last_pos = np.reshape(last_pos, [-1, 14])
        pyflex.step()
        cur_pos = np.array(pyflex.get_shape_states())
        cur_pos = np.reshape(cur_pos, [-1, 14])
        print("action: {}".format(action))
        action = action.reshape([-1, 3])
        #action = np.hstack([action, np.zeros([action.shape[0], 1])])
        vels = pyflex.get_velocities()
        #cur_pos[:, 3] = 1
        if self.action_mode == 'key_point_pos':
            cur_pos[0][0:3] = last_pos[0][0:3]+action[0, :]
            cur_pos[0][3:6] = last_pos[0][0:3]
            cur_pos[0][1] = max(cur_pos[0][1], 0.06)
            cur_pos[1][0:3] = last_pos[1][0:3] + action[1, :]
            cur_pos[1][3:6] = last_pos[1][0:3]
            cur_pos[1][1] = max(cur_pos[1][1], 0.06)

        pyflex.set_shape_states(cur_pos.flatten())
        #pyflex.set_velocities(vels.flatten())
        obs = self._get_obs()
        reward = self.compute_reward()
        return obs, reward, False, {}

    def compute_reward(self):
        
        calculate by taking max x,y cood and min x,y coord, create a discritized grid between
        the points
        :param pos:
        :return:
        
        pos = pyflex.get_positions()
        pos = np.reshape(pos, [-1, 4])
        print("Pos xs: {}".format(pos[:, 0]))
        minX = np.min(pos[:, 0])
        minY = np.min(pos[:, 2])
        maxX = np.max(pos[:, 0])
        maxY = np.max(pos[:, 2])
        grid = np.zeros([101, 101])
        init = np.array([minX, minY])
        span = np.array([maxX - minX, maxY-minY])/100.
        print("init: {}".format(init))
        pos2d = pos[:, [0, 2]]
        offset = pos2d - init
        slottedX = (offset[:, 0]//span[0])
        slottedy = (offset[:, 1]//span[1])
        grid[slottedy.astype(int),slottedX.astype(int)] = 1
        
        for i in range(len(pos2d)):
            offset = pos2d[i] - init
            print("offset: {} span: {}".format(offset, span))
            slottedX = int(offset[0]/span[0])
            slottedY = int(offset[1]/span[1])
            grid[slottedY,slottedX] = 1
        
        return np.sum(np.sum(grid))*span[0]*span[1]

    def set_scene(self):
        scene_params = np.array([0, 64, 32])
        pyflex.set_scene(10, scene_params, 0)
"""

if __name__ == "__main__":
    env = ClothFlattenPointControlEnv('key_point', 'sphere')
    env.reset(dropPoint=100)
    print("reset, entering loop")
    haveGrasped = False
    for i in range(0, 700):
        if env.prev_middle[0, 1] > 0.11 and not haveGrasped:
            obs, reward, _, _ = env.step(np.array([0., -0.001, 0, 0, 0.01] * 2))
            print("reward: {}".format(reward))
        elif not haveGrasped:
            obs, reward, _, _ = env.step(np.array([0., -0.001, 0, 0, -0.01] * 2))
            print("reward: {}".format(reward))
            if env.prev_dist[0] < 0.21:
                haveGrasped = True
        else:
            obs, reward, _, _ = env.step(np.array([0., 0.001, 0, 0, 0] * 2))
            print("reward: {}".format(reward))
