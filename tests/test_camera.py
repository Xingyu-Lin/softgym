import gym
import numpy as np
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt


def test_picker():
    num_picker = 3
    env = ClothFlattenEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth')

    for _ in range(10):
        env.reset()

        for i in range(50):
            print('step: ', i)
            action = np.zeros((num_picker, 4))
            if i < 12:
                action[:, 1] = -0.01
                action[:, 3] = 0
            elif i < 30:
                action[:, 1] = 0.01
                action[:, 3] = 1
            elif i < 40:
                action[:, 3] = 0
            env.step(action)


def test_random(env, N=5):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    # test_picker()
    num_picker = 200
    env = ClothFlattenEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=num_picker,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='cloth',
        num_variations=200,
        use_cached_states=True,
        deterministic=False)
    # exit()
    test_random(env)
