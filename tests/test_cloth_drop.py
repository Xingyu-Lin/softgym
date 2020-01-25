import gym
import numpy as np
import pyflex
from softgym.envs.cloth_drop import ClothDropEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt


def test_picker(env, num_picker=2):
    for _ in range(10):
        env.reset()

        for i in range(50):
            print('step: ', i)
            action = np.zeros((num_picker, 4))
            if i < 10:
                action[:, 0] = 0.02
                action[:, 3] = 1.
            elif i < 15:
                action[:, 0] = -0.02
                action[:, 3] = 1.
            elif i < 40:
                action[:, 3] = 0.
            obs, reward, _, _ = env.step(action)
            print(reward)


def test_random(env, N=5):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            action = action.reshape(-1, 4)
            _, reward, _, _ = env.step(action.flatten())
            print(reward)


if __name__ == '__main__':
    num_picker = 2
    env = ClothDropEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=50,
        action_repeat=8,
        render_mode='cloth',
        num_variations=200,
        use_cached_states=True,
        deterministic=False)
    # test_random(env)
    test_picker(env)
