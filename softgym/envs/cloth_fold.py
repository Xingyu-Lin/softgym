import numpy as np
import os
import os.path as osp
from gym.spaces import Box

import pyflex
from softgym.envs.cloth_env import ClothEnv


class ClothFoldPointControlEnv(ClothEnv):
    def __init__(self, observation_mode, action_mode, horizon=200, **kwargs):
        self.cloth_xdim = 64
        self.cloth_ydim = 32
        self.action_mode = action_mode
        self.observation_mode = observation_mode

        super().__init__(config_file="ClothFoldConfig.yaml", **kwargs)
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point', 'sphere', 'force', 'sticky', 'block']

        self.horizon = horizon

        if observation_mode == 'key_point':
            self.observation_space = Box(np.array([-np.inf] * 6), np.array([np.inf] * 6), dtype=np.float32)
            self.obs_key_point_idx = self.get_obs_key_point_idx()
        else:
            raise NotImplementedError

        if action_mode == 'key_point':
            self.action_space = Box(np.array([-1.] * 6), np.array([1.] * 6), dtype=np.float32)
            self.action_key_point_idx = self.get_action_key_point_idx()
        elif action_mode == 'sphere' or action_mode == 'block':
            space_low = np.array([-0.1, -0.1, -0.1, -0.1] * 2)
            space_high = np.array([0.1, 0.1, 0.1, 0.1] * 2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode == 'force':
            space_low = np.array([-0.1, -0.1, -0.1, -0.1, -0.1]*2)
            space_high = np.array([0.1, 0.1, 0.1, 0.1, 0.1]*2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode == 'sticky':
            space_low = (np.array([-0.1, -0.1, -0.1, -0.1]))
            space_high = (np.array([0.1, 0.1, 0.1, 0.1]))
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        else:
            raise NotImplementedError

        self.init_state = self.get_state()
        self.init_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim

    def get_obs_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = self.cloth_xdim * (self.cloth_ydim - 1)
        return np.array([idx_p1, idx_p2])

    def get_action_key_point_idx(self):
        idx_p1 = 0
        idx_p2 = self.cloth_xdim * (self.cloth_ydim - 1)
        return np.array([idx_p1, idx_p2])

    def set_scene(self):
        '''
        Setup the cloth scene and split particles into two groups for folding
        :return:
        '''
        # Set camera parameters.
        camera_x, camera_y, camera_z = self.camera_params['pos'][0], \
                                       self.camera_params['pos'][1], \
                                       self.camera_params['pos'][2]
        camera_ax, camera_ay, camera_az = self.camera_params['angle'][0], \
                                          self.camera_params['angle'][1], \
                                          self.camera_params['angle'][2]
        scene_params = [self.cloth_xdim, self.cloth_ydim]
        scene_params.extend(
            [camera_x, camera_y, camera_z, camera_ax, camera_ay, camera_az, self.camera_width, self.camera_height])

        self.scene_params = np.array(scene_params)
        # pyflex.set_scene(9, self.scene_params, 0)
        super().set_scene(sizex=self.cloth_xdim, sizey=self.cloth_ydim)
        # Set folding group
        particle_grid_idx = np.array(list(range(self.cloth_xdim * self.cloth_ydim))).reshape(self.cloth_ydim,
                                                                                             self.cloth_xdim)

        x_split = self.cloth_xdim // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = particle_grid_idx[:, self.cloth_xdim:x_split - 1:-1].flatten()

        colors = np.zeros([self.cloth_ydim * self.cloth_xdim])
        colors[self.fold_group_b] = 1

        self.set_colors(colors)
        if self.action_mode == 'sphere':
            self.add_spheres()
        if self.action_mode == 'force':
            self.addForce()
        if self.action_mode == 'sticky':
            self.addSticky()
        if self.action_mode == 'block':
            self.addBlocks()
        # self.set_test_color()
        print("scene set")
    def set_test_color(self):
        '''
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        '''
        colors = np.zeros((self.cloth_xdim * self.cloth_ydim))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def get_current_observation(self):
        pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        return pos[self.obs_key_point_idx, :3].flatten()
    #def get_sticky(self):
    #    super().get_sticky()
    def reset(self):
        self.time_step = 0
        self.set_state(self.init_state)
        if self.action_mode == 'sphere' or self.action_mode == 'block':
            self.sphere_reset()
        if self.action_mode == 'force':
            self.force_reset()
        # for _ in range(100):
        #pyflex.step()
        return self.get_current_observation()

    def compute_reward(self, pos):
        '''
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        '''
        pos_group_a = pos[self.fold_group_a, :3]
        pos_group_b = pos[self.fold_group_b, :3]
        pos_group_b_init = self.init_pos[self.fold_group_b, :3]
        distance = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        distance_to_init = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))

        #return np.max(pos[:, 1])
        return -distance - distance_to_init

    def _step(self, action):
        lastPos = pyflex.get_shape_states()
        pyflex.step()
        self.time_step += 1
        if self.action_mode == 'key_point':
            action[2] = 0
            action[5] = 0
            action = np.array(action) / 10.
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            action = action.reshape([-1, 3])
            action = np.hstack([action, np.zeros([action.shape[0], 1])])
            cur_pos[self.action_key_point_idx, :] = last_pos[self.action_key_point_idx] + action
            pyflex.set_positions(cur_pos.flatten())
            reward = self.compute_reward(cur_pos)
        elif self.action_mode == 'force':
            self.forceStep(action)
            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            reward = self.compute_reward(cur_pos)
        elif self.action_mode == 'sticky':
            self.sticky_step(action)
            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            reward = self.compute_reward(cur_pos)
        elif self.action_mode == 'block':
            self.boxStep(action, lastPos)
            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            reward = self.compute_reward(cur_pos)
        else:
            self.sphereStep(action, lastPos)
            cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            reward = self.compute_reward(cur_pos)
        obs = self.get_current_observation()

        return obs, reward, False, {}

    def set_video_recording_params(self):
        """
        Set the following parameters if video recording is needed:
            video_idx_st, video_idx_en, video_height, video_width
        """
        self.video_height = 240
        self.video_width = 320
