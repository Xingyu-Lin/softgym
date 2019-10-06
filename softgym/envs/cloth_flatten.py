import numpy as np
from gym.spaces import Box
import random

import pyflex
from softgym.envs.flex_env import FlexEnv


class ClothFlattenPointControlEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode):
        super().__init__()
        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb']
        assert action_mode in ['key_point']

        self.observation_mode = observation_mode
        self.action_mode = action_mode

        if observation_mode == 'key_point':
            self.observation_space = Box(np.array([-np.inf] * 6), np.array([np.inf] * 6), dtype=np.float32)
        else:
            raise NotImplementedError

        if action_mode == 'key_point':
            self.action_space = Box(np.array([-1.] * 6), np.array([1.] * 6), dtype=np.float32)
        else:
            raise NotImplementedError
        self.time_step = 500
        self.init_state = self.get_state()

    def reset(self, dropPoint = None):

        pickpoint = random.randint(0, 64 * 32)
        if dropPoint is not None:
            pickpoint = dropPoint
        pyflex.set_scene(11, np.array([pickpoint]), 0)

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
                vels[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            stopped = True
            for j in range(pyflex.get_n_particles()):
                if vels[j] > 0.2:
                    print("stopped check at {} with vel {}".format(j, vels[j]))
                    stopped = False
                    break
            if stopped:
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

    def step(self, action):
        last_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        pyflex.step()
        cur_pos = np.array(pyflex.get_positions()).reshape([-1, 4])
        action = action.reshape([-1, 4])
        idxs = np.hstack(action[:, 0])
        updates = action[:, 1:]
        action = np.hstack([action, np.zeros([action.shape[0], 1])])
        cur_pos[idxs.astype(int), :3] = last_pos[idxs.astype(int)][:, :3] + updates
        pyflex.set_positions(cur_pos.flatten())
        obs = self.get_current_observation()
        reward = self.compute_reward(cur_pos)
        return obs, reward, False, {}

    def compute_reward(self, pos):
        return 0.

    def set_scene(self):
        scene_params = np.array([0])
        pyflex.set_scene(10, scene_params, 0)

if __name__ == "__main__":
    pyflex.init()
    env = ClothFlattenPointControlEnv('key_point', 'key_point')
    env.reset()
    print("reset, entering loop")
    for i in range(0, 500):
        env.step(np.array([150, 0, 0.01, 0,]))

