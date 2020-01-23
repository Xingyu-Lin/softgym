import gym
import numpy as np
import pyflex
from softgym.envs.dough_flatten import DoughFlattenEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt


def test_random(env):
    N = 10
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(20):
            action = np.array([0, -0.002, 0.002, 0])
            action = env.action_space.sample()
            env.step(action)
        # for _ in range(50):
        #     action = np.array([0, 0.05, 0, 0])
        #     env.step(action)


if __name__ == '__main__':
    # test_picker()
    env = DoughFlattenEnv(
        observation_mode='cam_rgb',
        action_mode='direct',
        render=True,
        headless=False,
        horizon=75,
        action_repeat=8,
        render_mode='dough',
        num_variations=2,
        use_cached_states=True,
        deterministic=False)
    test_random(env)
