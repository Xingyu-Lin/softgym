import gym
import softgym
import numpy as np
from softgym.envs.cloth_fold import ClothFoldPointControlEnv
if __name__ == '__main__':
    # env = ClothFoldPointControlEnv(observation_mode='key_point', action_mode='key_point')
    softgym.register_flex_envs()
    env = gym.make('ClothFoldSphereControl-v0')
    haveGrasped = False
    while (1):
        env.reset()
        for i in range(350):
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
