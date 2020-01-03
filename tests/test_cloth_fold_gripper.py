import gym
import softgym
import numpy as np
from softgym.envs.cloth_fold import ClothFoldPointControlEnv
if __name__ == '__main__':
    # env = ClothFoldPointControlEnv(observation_mode='key_point', action_mode='key_point')
    softgym.register_flex_envs()
    env = gym.make('ClothFoldBoxControl-v0')
    haveGrasped = False
    while (1):
        env.reset()
        haveGrasped = False
        for i in range(500):
            if env.prev_middle[0,1] > 0.05 and not haveGrasped:
                obs, reward, _, _ = env.step(np.array([0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0]))
            if env.prev_dist[0] > 0.12 and not haveGrasped:
                obs, reward, _, _ = env.step(np.array([0, 0, 0, 0, -0.001]*2))
            else:
                haveGrasped = True
                obs, reward, _, _ = env.step(np.array([0, 0.005, 0, 0, 0]*2))