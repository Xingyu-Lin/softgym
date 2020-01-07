import gym
import softgym
import numpy as np
from softgym.envs.cloth_fold import ClothFoldEnv
if __name__ == '__main__':
    # env = ClothFoldPointControlEnv(observation_mode='key_point', action_mode='key_point')
    softgym.register_flex_envs()
    env = gym.make('ClothFoldSphereControl-v0')
    while (1):
        env.reset()
        for i in range(20):
            # action = np.array([0.4, 0., 0., 0.4, 0., 0.])
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print('step: {}, reward: {}'.format(i, reward))
            env.render(mode='rgb_array')
