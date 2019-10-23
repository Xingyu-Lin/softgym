import gym
import softgym
import numpy as np
from softgym.envs.cloth_fold import ClothFoldPointControlEnv
if __name__ == '__main__':
    # env = ClothFoldPointControlEnv(observation_mode='key_point', action_mode='key_point')
    softgym.register_flex_envs()
    env = gym.make('ClothFoldPointControl-v0')
    while (1):
        env.reset()
        for i in range(150):
            action = np.array([0.4, 0., 0., 0.4, 0., 0.])
            observation, reward, done, info = env.step(action)
            print('step: {}, reward: {}'.format(i, reward))
            env.render(mode='rgb_array')
