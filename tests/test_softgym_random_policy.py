import gym
import softgym
from softgym.envs.cloth_fold import ClothFoldPointControlEnv
if __name__ == '__main__':
    # env = ClothFoldPointControlEnv(observation_mode='key_point', action_mode='key_point')
    softgym.register_flex_envs()
    env = gym.make('ClothFoldPointControl-v0')
    while (1):
        env.reset()
        for _ in range(300):
            action = env.action_space.sample() / 50.
            env.step(action)
