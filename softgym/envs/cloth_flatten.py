import numpy as np
from gym.spaces import Box
import random
import os
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv


class ClothFlattenPointControlEnv(ClothEnv):
    def __init__(self, observation_mode, action_mode, horizon=250, **kwargs):
        super().__init__(config_file="ClothFlattenConfig.yaml", **kwargs)
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point_pos', 'key_point_vel', 'sphere']

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.video_path = "data/videos"
        self.prev_reward = 0

        self.horizon = horizon

        if observation_mode == 'key_point':  # TODO: add sphere position
            if action_mode == 'key_point_pos':
                self.observation_space = Box(np.array([-np.inf] * pyflex.get_n_particles() * 3),
                                             np.array([np.inf] * pyflex.get_n_particles() * 3), dtype=np.float32)
            elif action_mode == 'sphere':
                self.observation_space = Box(np.array([-np.inf] * (pyflex.get_n_particles() * 3 + 4 * 3)),
                                             np.array([np.inf] * (pyflex.get_n_particles() * 3 + 4 * 3)), dtype=np.float32)
        else:
            raise NotImplementedError

        if action_mode.startswith('key_point'):
            space_low = np.array([0, -0.1, -0.1, -0.1] * 2)
            space_high = np.array([3.9, 0.1, 0.1, 0.1] * 2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode.startswith('sphere'):
            space_low = np.array([-0.1, -0.1, -0.1, -0.1, -0.1] * 2) * 0.1
            space_high = np.array([0.1, 0.1, 0.1, 0.1, 0.1] * 2) * 0.1

            self.action_space = Box(space_low, space_high, dtype=np.float32)

        self.reset_time_step = 300
        self.init_state = self.get_state()
        self.storage_name = "test_flatten"
        self.i = 0  # actual step without counting action repetation
        self.time_step = 0  # step with action repetation
        self.initPos = None
        self.initVel = None
        self.video_height = 240
        self.video_width = 320

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

    def reset(self, dropPoint=1000):
        self.i = 0
        self.time_step = 0
        if self.initPos is not None:
            pyflex.set_positions(self.initPos)
            pyflex.set_velocities(self.initVel)
            pyflex.set_shape_states(self.initState)
            if self.action_mode == 'sphere':
                self.sphere_reset()
            pyflex.step()
            return self._get_obs()

        pickpoint = random.randint(0, self.xdim * self.ydim)
        if dropPoint is not None:
            pickpoint = dropPoint
        firstPos = pyflex.get_positions()
        firstPos[pickpoint * 4 + 3] = 0
        # print("{}".format(firstPos[pickpoint * 4: pickpoint * 4 + 3]))
        pyflex.set_positions(firstPos)
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)
        # print("pick point: {}".format(pickpoint))
        particle_pos = pyflex.get_positions()[pickpoint * 4: pickpoint * 4 + 3]
        stopParticle = True
        for i in range(0, self.reset_time_step):
            pyflex.step()
            newPos = pyflex.get_positions()
            vels = pyflex.get_velocities()
            if stopParticle:
                newPos[pickpoint * 4: pickpoint * 4 + 3] = particle_pos
                # print("pick pos: {}".format(newPos[pickpoint * 4 + 2]))
                # newPos[pickpoint * 4 + 3] = 0.0
                vels[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            stopped = True
            for j in range(pyflex.get_n_particles()):
                if vels[j] > 0.01:
                    # print("stopped check at {} with vel {}".format(j, vels[j]))
                    stopped = False
                    break
            if stopped:
                newPos[pickpoint * 4 + 3] = 1
                stopParticle = False

            pyflex.set_velocities(vels)
            pyflex.set_positions(newPos)
        print("dropping")
        lastPos = pyflex.get_positions()
        lastPos[pickpoint * 4 + 3] = 1
        pyflex.set_positions(lastPos)
        for i in range(0, 100):
            pyflex.step()
        if self.action_mode.startswith('sphere'):
            super().addSpheres(pick_point=pickpoint)  # YF: should add sphere near the drop point
        self.initPos = pyflex.get_positions()
        self.initVel = pyflex.get_velocities()
        self.initState = pyflex.get_shape_states()
        self.prev_reward = self.compute_reward()
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
        # print("stepping")
        self.i = self.i + 1
        if self.action_mode.startswith('key_point'):
            valid_idxs = np.array([0, 63, 31 * 64, 32 * 64 - 1])
            last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
            des_dir = self.storage_name
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
            # print("sphering")
            last_pos = pyflex.get_shape_states()
            if self.record_video:  # NOTE: just add a flag
                pyflex.step(capture=1, path=os.path.join(self.video_path, 'render_{}.tga'.format(self.i)))
            else:
                pyflex.step()
            super().sphereStep(action, last_pos)
        # print("computing reward")
        obs = self._get_obs()
        reward = self.compute_reward(action, obs)
        # TODO Xingyu Change the compute_reward function
        dreward = reward - self.prev_reward
        self.prev_reward = reward
        # print("returning")
        return obs, dreward, False, {}

    def compute_reward(self, obs=None, action=None):
        """
        calculate by taking max x,y cood and min x,y coord, create a discritized grid between
        the points
        :param pos:
        :return:
        """
        pos = pyflex.get_positions()
        pos = np.reshape(pos, [-1, 4])
        minX = np.min(pos[:, 0])
        minY = np.min(pos[:, 2])
        maxX = np.max(pos[:, 0])
        maxY = np.max(pos[:, 2])
        grid = np.zeros([101, 101])
        init = np.array([minX, minY])
        span = np.array([maxX - minX, maxY - minY]) / 100.
        pos2d = pos[:, [0, 2]]
        offset = pos2d - init
        slottedX = (offset[:, 0] // span[0])
        slottedy = (offset[:, 1] // span[1])
        grid[slottedy.astype(int), slottedX.astype(int)] = 1
        """
        for i in range(len(pos2d)):
            offset = pos2d[i] - init
            print("offset: {} span: {}".format(offset, span))
            slottedX = int(offset[0]/span[0])
            slottedY = int(offset[1]/span[1])
            grid[slottedY,slottedX] = 1
        """
        return np.sum(np.sum(grid)) * span[0] * span[1]

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
