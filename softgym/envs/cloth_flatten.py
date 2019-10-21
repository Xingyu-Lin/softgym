import numpy as np
from gym.spaces import Box
import random
import os
import pyflex
from softgym.envs.flex_env import FlexEnv


class ClothFlattenPointControlEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode):
        super().__init__()
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point_pos', 'key_point_vel']

        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.video_path = "data/videos"

        self.horizon = 100

        if observation_mode == 'key_point':
            self.observation_space = Box(np.array([-np.inf] * pyflex.get_n_particles()), np.array([np.inf] * pyflex.get_n_particles()), dtype=np.float32)
        else:
            raise NotImplementedError

        if action_mode.startswith('key_point'):
            space_low = np.array([0, -0.1, -0.1, -0.1]*2)
            space_high = np.array([3.9, 0.1, 0.1, 0.1]*2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        else:
            raise NotImplementedError
        self.time_step = 500
        self.init_state = self.get_state()
        self.storage_name = "test_flatten"
        self.i = 0
        self.initPos = None
        self.initVel = None
        self.video_height = 240
        self.video_width = 320

    def reset(self, dropPoint = 1000, xdim=64, ydim=32):
        self.i=0
        if self.initPos is not None:
            print("resetting")
            pyflex.set_positions(self.initPos)
            pyflex.set_positions(self.initVel)
            return

        pickpoint = random.randint(0, xdim * ydim)
        if dropPoint is not None:
            pickpoint = dropPoint
        pyflex.set_scene(10, np.array([pickpoint, xdim, ydim]), 0)

        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)
        print("pick point: {}".format(pickpoint))
        particle_pos = pyflex.get_positions()[pickpoint * 4: pickpoint * 4 + 3]
        stopParticle = True
        for i in range(0, self.time_step):
            pyflex.step()
            newPos = pyflex.get_positions()
            vels = pyflex.get_velocities()
            if stopParticle:
                newPos[pickpoint * 4: pickpoint * 4 + 3] = particle_pos
                #newPos[pickpoint * 4 + 3] = 0.0
                vels[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            stopped = True
            for j in range(pyflex.get_n_particles()):
                if vels[j] > 0.01:
                    #print("stopped check at {} with vel {}".format(j, vels[j]))
                    stopped = False
                    break
            if stopped:
                newPos[pickpoint * 4 + 3] = 1
                stopParticle = False

            pyflex.set_velocities(vels)
            pyflex.set_positions(newPos)
        self.initPos = pyflex.get_positions()
        self.initVel = pyflex.get_velocities()

        #pyflex.set_scene(9, np.array([]), 0)
        #pyflex.set_positions(self.initPos)

    def get_current_observation(self):
        pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        return pos[:, :3].flatten()

    def set_video_recording_params(self):
        """
        Set the following parameters if video recording is needed:
            video_idx_st, video_idx_en, video_height, video_width
        """
        self.video_height = 240
        self.video_width = 320

    def step(self, action):
        print("stepping")
        valid_idxs = np.array([0, 63, 31*64, 32*64-1])
        last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        des_dir = self.storage_name
        pyflex.step(capture=1, path=os.path.join(self.video_path, 'render_{}.tga'.format(self.i)))
        self.i = self.i+1
        cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        action = action.reshape([-1, 4])
        idxs = np.hstack(action[:, 0])
        print("idxs: {}".format(valid_idxs[idxs.astype(int)]))
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
        print("computing reward")
        obs = self.get_current_observation()
        reward = self.compute_reward()
        print("returning")
        return obs, reward, False, {}

    def compute_reward(self):
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
        span = np.array([maxX - minX, maxY-minY])/100.
        pos2d = pos[:, [0, 2]]
        offset = pos2d - init
        slottedX = (offset[:, 0]//span[0])
        slottedy = (offset[:, 1]//span[1])
        grid[slottedy.astype(int),slottedX.astype(int)] = 1
        """
        for i in range(len(pos2d)):
            offset = pos2d[i] - init
            print("offset: {} span: {}".format(offset, span))
            slottedX = int(offset[0]/span[0])
            slottedY = int(offset[1]/span[1])
            grid[slottedY,slottedX] = 1
        """
        return np.sum(np.sum(grid))*span[0]*span[1]

    def set_scene(self):
        scene_params = np.array([0, 64, 32])
        pyflex.set_scene(10, scene_params, 0)

class ClothFlattenSphereControlEnv(FlexEnv):
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
        self.time_step = 500
        self.init_state = self.get_state()

    def reset(self, dropPoint = None, xdmin = 64, ydim = 32):

        pickpoint = random.randint(0, xdmin * ydim)
        if dropPoint is not None:
            pickpoint = dropPoint
        pyflex.set_scene(10, np.array([pickpoint, xdmin, ydim]), 0)

        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)
        print("pick point: {}".format(pickpoint))
        particle_pos = pyflex.get_positions()[pickpoint * 4: pickpoint * 4 + 3]
        stopParticle = True
        for i in range(0, self.time_step):
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
        self.initPos = pyflex.get_positions()
        self.initVel = pyflex.get_velocities()
        pyflex.add_sphere(0.1, [0.5, 0.5, 0], [1, 0, 0, 0])
        pyflex.add_sphere(0.1, [-0.5, 0.5, 0], [1, 0, 0, 0])
        #pyflex.set_scene(9, np.array([]), 0)
        #pyflex.set_positions(self.initPos)

    def get_current_observation(self):
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
        #action = action.reshape([-1, 3])
        #action = np.hstack([action, np.zeros([action.shape[0], 1])])
        vels = pyflex.get_velocities()
        #cur_pos[:, 3] = 1
        if self.action_mode == 'key_point_pos':
            cur_pos[0][0:3] = last_pos[0][0:3]+action[0]
            cur_pos[0][3:6] = last_pos[0][0:3]
            cur_pos[1][0:3] = last_pos[1][0:3] + action[1]
            cur_pos[1][3:6] = last_pos[1][0:3]
        pyflex.set_shape_states(cur_pos.flatten())
        #pyflex.set_velocities(vels.flatten())
        obs = self.get_current_observation()
        reward = self.compute_reward()
        return obs, reward, False, {}

    def compute_reward(self):
        """
        calculate by taking max x,y cood and min x,y coord, create a discritized grid between
        the points
        :param pos:
        :return:
        """
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
        """
        for i in range(len(pos2d)):
            offset = pos2d[i] - init
            print("offset: {} span: {}".format(offset, span))
            slottedX = int(offset[0]/span[0])
            slottedY = int(offset[1]/span[1])
            grid[slottedY,slottedX] = 1
        """
        return np.sum(np.sum(grid))*span[0]*span[1]

    def set_scene(self):
        scene_params = np.array([0, 64, 32])
        pyflex.set_scene(10, scene_params, 0)

if __name__ == "__main__":
    pyflex.init()

    env = ClothFlattenSphereControlEnv('key_point', 'key_point_pos')
    des_dir = env.storage_name
    os.system('mkdir -p ' + des_dir)
    env.reset(dropPoint=100)
    print("reset, entering loop")
    for i in range(0, 500):
        print("going up: {}",format(i))
        obs, reward, _, _ = env.step(np.array([[-0.003, 0, -0.003], [0.003, 0, 0.003]]))
        print("reward: {}".format(reward))

